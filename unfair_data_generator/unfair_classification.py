import numpy as np
from sklearn.utils import check_random_state, shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement

try:
    from util.helpers import _generate_hypercube, _generate_leaky_variables, get_params_for_certain_equality_type
except ImportError:
    from unfair_data_generator.util.helpers import _generate_hypercube, _generate_leaky_variables, get_params_for_certain_equality_type


def make_unfair_classification(
    n_samples=100,
    n_features=20,
    *,
    n_informative=2,
    n_redundant=2,
    n_repeated=0,
    n_leaky=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=None,
    group_params=None,
    n_sensitive_groups=2,
    fairness_type="Equal quality",
    return_sensitive_group_centroids=False,
    append_sensitive_to_X=False
):
    """Generate a random n-class classification problem.

    This function extends scikit-learn's make_classification to include sensitive
    group information and fairness constraints. It creates clusters of points 
    normally distributed (std=1) about vertices of an ``n_informative``-dimensional 
    hypercube with sides of length ``2*class_sep`` and assigns samples to sensitive
    groups based on specified fairness criteria.

    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.

    Parameters:
        n_samples (int, default=100): The number of samples.
        n_features (int, default=20): The total number of features. These comprise ``n_informative`` informative features, ``n_redundant`` redundant features, ``n_repeated`` duplicated features and ``n_features-n_informative-n_redundant-n_repeated`` useless features drawn at random.
        n_informative (int, default=2): The number of informative features. Each class is composed of a number of gaussian clusters each located around the vertices of a hypercube in a subspace of dimension ``n_informative``. For each cluster, informative features are drawn independently from  N(0, 1) and then randomly linearly combined within each cluster in order to add covariance. The clusters are then placed on the vertices of the hypercube.
        n_redundant (int, default=2): The number of redundant features. These features are generated as random linear combinations of the informative features.
        n_repeated (int, default=0): The number of duplicated features, drawn randomly from the informative and the redundant features.
        n_leaky (int, default=0): The number of leaky features. These features are generated as combinations of the sensitive features.
        n_classes (int, default=2): The number of classes (or labels) of the classification problem.
        n_clusters_per_class (int, default=2): The number of clusters per class.
        weights : (array-like of shape (n_classes,) or (n_classes - 1,), default=None): The proportions of samples assigned to each class. If None, then classes are balanced. Note that if ``len(weights) == n_classes - 1``, then the last class weight is automatically inferred. More than ``n_samples`` samples may be returned if the sum of ``weights`` exceeds 1. Note that the actual class proportions will not exactly match ``weights`` when ``flip_y`` isn't 0.
        flip_y (float, default=0.01): The fraction of samples whose class is assigned randomly. Larger values introduce noise in the labels and make the classification task harder. Note that the default setting flip_y > 0 might lead to less than ``n_classes`` in y in some cases.
        class_sep (float, default=1.0): The factor multiplying the hypercube size.  Larger values spread out the clusters/classes and make the classification task easier.
        hypercube (bool, default=True): If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.
        shift (float, ndarray of shape (n_features,) or None, default=0.0): Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].
        scale (float, ndarray of shape (n_features,) or None, default=1.0): Multiply features by the specified value. If None, then features are scaled by a random value drawn in [1, 100]. Note that scaling happens after shifting.
        shuffle (bool, default=True): Shuffle the samples and the features.
        random_state (int, RandomState instance or None, default=None): Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls.
        group_params (dict, default=None): A dictionary specifying group-specific parameters for creating clusters and assigning samples to sensitive groups. Each key in the dictionary represents a sensitive group, and its value is another dictionary that may include:

            - 'class_sep' (float): The separation factor for clusters belonging to the group. Larger values create better-separated clusters.

            - 'weights' (array-like of shape (n_classes,)): The proportion of samples assigned to each class within the group. If None, group_params will be automatically generated using fairness_type and n_sensitive_groups.

        n_sensitive_groups (int, default=2): The number of sensitive groups to create. Must be between 2 and 5. Used to automatically generate group_params when group_params is None.
        fairness_type (str, default="Equal quality"): The type of fairness/equality to simulate when automatically generating group_params. Used only when group_params is None. Supported values:

            - "Equal quality"

            - "Demographic parity"

            - "Equal opportunity"

            - "Equalized odds"

        return_sensitive_group_centroids (bool, default=False): If True, return the group centroids as an additional output. These centroids represent the centers of the clusters used to generate te data for each sensitive group and can be useful for visualization and analysis.
        append_sensitive_to_X (bool, default=False): If True, append the sensitive group information as an additional column to X. This adds the sensitive group labels as the last feature column in the returned X matrix. If False, the sensitive group information is returned as a separate array Z.

    Returns:
        tuple: A tuple containing the following elements:

            - X (ndarray): The generated samples. Shape (n_samples, n_features).
            - y (ndarray): The integer labels for class membership of each sample. Shape (n_samples,).
            - Z (ndarray, optional): Only returned when append_sensitive_to_X=False. Contains integer group identifiers where each unique value represents a different sensitive group. Shape (n_samples,).
            - group_centroids (dict, optional): Dictionary of group-specific centroids. Only returned when return_sensitive_group_centroids=True.

    Note:
        The function returns different combinations of outputs based on parameters:

            - append_sensitive_to_X=False, return_sensitive_group_centroids=False: (X, y, Z)

            - append_sensitive_to_X=False, return_sensitive_group_centroids=True: (X, y, Z, group_centroids)  

            - append_sensitive_to_X=True, return_sensitive_group_centroids=False: (X, y)

            - append_sensitive_to_X=True, return_sensitive_group_centroids=True: (X, y, group_centroids)
    """

    generator = check_random_state(random_state)

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated + n_leaky > n_features:
        raise ValueError(
            "Number of informative, redundant and repeated "
            "features must sum to less than the number of total"
            f" features ({n_features})"
        )
    if n_sensitive_groups < 2 or n_sensitive_groups > 5:
        raise ValueError(
            f"Number of sensitive groups must be between 2 and 5, got {n_sensitive_groups}"
        )
    # Use log2 to avoid overflow errors
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        msg = "n_classes({}) * n_clusters_per_class({}) must be"
        msg += " smaller or equal 2**n_informative({})={}"
        raise ValueError(
            msg.format(n_classes, n_clusters_per_class,
                       n_informative, 2**n_informative)
        )

    # Build group-specific weights
    if group_params is None:
        # Generate group_params using fairness_type and n_sensitive_groups
        group_params = get_params_for_certain_equality_type(
            fairness_type, n_sensitive_groups)

    if group_params is not None:
        group_weights = {}
        for group, params in group_params.items():
            # Extract weights for the group
            group_weights[group] = params.get("weights", [0.5, 0.5])
        n_groups = len(group_params)
    else:
        # Fallback: Default behavior when group_params is still None
        group_weights = {"default": [0.5, 0.5]}
        group_params = {"default": {}}
        n_groups = 1

    # Calculate remaining feature counts
    n_useless = n_features - n_informative - n_redundant - n_repeated - n_leaky

    # Multiply cluster amount by amount of sensitive groups
    n_clusters = n_classes * n_clusters_per_class * n_groups

    # Distribute samples among clusters based on group-specific weights
    n_samples_per_cluster = []
    for group, weights in group_weights.items():
        # Equal split of samples per group
        group_sample_count = n_samples // n_groups
        for class_idx in range(n_classes):
            class_sample_count = int(group_sample_count * weights[class_idx])

            # Divide samples equally among clusters for this class within the group
            divider = n_clusters_per_class
            class_sample_count_per_cluster = class_sample_count // divider

            for cluster_idx in range(divider):
                n_samples_per_cluster.append(class_sample_count_per_cluster)

    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X and y
    X = np.zeros((n_samples, n_features - 1))
    # Initialize target labels
    y = np.zeros(n_samples, dtype=int)

    # Build group-specific centroids
    if group_params is not None:
        group_centroids = {}
        for group, params in group_params.items():
            group_class_sep = params.get(
                "class_sep", class_sep)  # Default to class_sep
            # Separate random state for groups
            group_generator = check_random_state(random_state)

            # Generate hypercube for group-specific centroids
            group_centroids[group] = _generate_hypercube(
                n_clusters // n_groups, n_informative, group_generator).astype(float)
            group_centroids[group] *= 2 * group_class_sep
            group_centroids[group] -= group_class_sep
    else:
        # Default centroids if no class_sep are specified
        group_centroids = {"default": _generate_hypercube(
            n_clusters, n_informative, generator).astype(float)}
        group_centroids["default"] *= 2 * class_sep
        group_centroids["default"] -= class_sep

    # Initially draw informative features from the standard normal
    X[:, :n_informative] = generator.standard_normal(
        size=(n_samples, n_informative))

    # Add a column to X to store the sensitive group
    # Initialize with zeros for the sensitive group column
    X = np.c_[X, np.zeros(n_samples)]
    n_features = n_features + 1

    # Assign samples to clusters and apply group-specific centroids
    stop = 0
    counter = 0
    # Pre-allocate array for sensitive group values
    sensitive_data = np.empty(n_samples, dtype=int)
    for i, group_name in enumerate(group_params.keys() if group_params is not None else []):
        for k in range(n_classes):
            for cluster_idx in range(n_clusters_per_class):
                start, stop = stop, stop + n_samples_per_cluster[counter]

                # Assign class labels based on the intended class distribution
                y[start:stop] = k

                # Get X values for this centroid
                X_k = X[start:stop, :n_informative]

                # Get the centroid
                centroid_index = k * n_clusters_per_class + cluster_idx
                centroid = group_centroids[group_name][centroid_index]

                A = np.eye(n_informative)  # Identity matrix (no covariance)
                X_k[...] = np.dot(X_k, A)
                X_k += centroid  # Shift to group-specific centroid

                # Assign sensitive group value to the new column in X
                # Set the sensitive group value to "i"
                sensitive_data[start:stop] = i

                counter = counter + 1

    # Create redundant features
    if n_redundant > 0:
        B = 2 * generator.uniform(size=(n_informative, n_redundant)) - 1
        X[:, n_informative: n_informative + n_redundant] = np.dot(
            X[:, :n_informative], B
        )

    # Generate repeated features (duplicates of existing informative/redundant features)
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.uniform(size=n_repeated) +
                   0.5).astype(np.intp)
        X[:, n: n + n_repeated] = X[:, indices]

    # Generate useless features
    if n_useless > 0:
        X[:, -
            n_useless:] = generator.standard_normal(size=(n_samples, n_useless))

    # Probability of correct prediction for each sensitive group
    sensitive_probabilities = {}
    for i, group in enumerate(group_params.keys() if group_params is not None else []):
        sensitive_probabilities[i] = np.random.uniform(0, 1)

    # Fill leaky features
    if n_leaky > 0:
        X_l = _generate_leaky_variables(
            sensitive_data, sensitive_probabilities, n_leaky)

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.uniform(size=n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * generator.uniform(size=n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * generator.uniform(size=n_features)
    X *= scale

    # Add leaky data to X
    if n_leaky > 0:
        X[:, -n_leaky:] = X_l

    # Handle sensitive data based on append_sensitive_to_X parameter
    if append_sensitive_to_X:
        X = np.column_stack((X, sensitive_data))
    else:
        Z = sensitive_data

    if shuffle:
        # Randomly permute samples
        if append_sensitive_to_X:
            indices = generator.permutation(n_samples)
            X, y = X[indices], y[indices]
        else:
            indices = generator.permutation(n_samples)
            X, y, Z = X[indices], y[indices], Z[indices]

    # Return based on parameters
    if append_sensitive_to_X:
        if return_sensitive_group_centroids:
            return X, y, group_centroids
        else:
            return X, y
    else:
        if return_sensitive_group_centroids:
            return X, y, Z, group_centroids
        else:
            return X, y, Z
