import numpy as np
import pandas as pd
from sklearn.utils.random import sample_without_replacement


def _generate_hypercube(samples, dimensions, rng):
    """
    Generate distinct binary samples of specified dimensions using hypercube sampling.

    This function creates a set of unique binary vectors by sampling from a hypercube
    space. For dimensions > 30, it uses a recursive approach to handle memory constraints.

    Args:
        samples (int): Number of distinct binary samples to generate.
        dimensions (int): Length of each binary sample (number of features).
        rng (numpy.random.RandomState): Random number generator for reproducible results.

    Returns:
        numpy.ndarray: Array of shape (samples, dimensions) containing binary values (0 or 1). Each row represents a unique binary sample.
    """

    if dimensions > 30:
        return np.hstack([
            rng.randint(2, size=(samples, dimensions - 30)),
            _generate_hypercube(samples, 30, rng),
        ])

    out = sample_without_replacement(2**dimensions, samples, random_state=rng).astype(
        dtype=">u4", copy=False
    )
    out = np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]
    return out


def _generate_leaky_variables(
    sensitive_data,
    sensitive_probabilities,
    n_leaky,
    return_weights=False,
    formula=False
):
    """
    Generates X variables based on given Z values and probabilities of correct predictions.

    Parameters:
        sensitive_data (np.array): Array of integer values representing categories (e.g., [0, 1, 2, 1, 0, 2]).
        sensitive_probabilities (dict): Dictionary mapping Z values to their prediction accuracy (0-1). Higher values mean the leaky variables will be more predictive of that group. Example: {0: 0.8, 1: 0.6, 2: 0.9} means category 0 has 80% accuracy rate, etc.
        n_leaky (int): Number of independent X variables to generate.
        return_weights (bool, default=False): If True, the function will return the weights used in the formula.
        formula (bool, default=False): If True, the function will print the formula used to generate the data.

    Returns:
        tuple: A tuple containing the following elements:
            - X_l (np.array): Generated leaky variables matrix. Each column represents one leaky feature, and each row corresponds to a sample from the input sensitive_data. np.array of shape (n_samples, n_leaky).
            - weights (dict, optional): Dictionary containing the weights used in the formula.
    """
    n_samples = len(sensitive_data)

    # Generate normal distribution for each leaky variable
    X_l = np.zeros((n_samples, n_leaky))
    for i in range(n_leaky):
        X_l[:, i] = np.random.normal(0.5, 0.1, n_samples)
    df = pd.DataFrame(
        X_l, columns=[f'L{i}' for i in range(n_leaky)])

    # Generate weights
    weights = {i: np.random.uniform(-1, 1) for i in range(n_leaky)}
    weights[-1] = np.random.uniform(-1, 1)

    # Calculate K
    df['K'] = weights[-1] + \
        sum(weights[i] * df[f'L{i}'] for i in range(n_leaky))
    df = df.sort_values(
        by='K', ascending=True).reset_index(drop=True)

    # Determine value intervals for each sensitive group
    df['sensitive'] = sensitive_data
    intervals = {}
    unique_groups = df['sensitive'].unique()
    for group in unique_groups:
        group_subset = df[df['sensitive'] == group]
        intervals[group] = {
            "min": group_subset["K"].min(),
            "max": group_subset["K"].max()
        }

    # Apply sensitive probabilities
    for group in unique_groups:
        group_subset = df[df['sensitive'] == group]
        n_samples = len(group_subset)
        n_samples_to_modify = int(
            n_samples * (1 - sensitive_probabilities[group]))
        modify_indices = np.random.choice(
            group_subset.index, n_samples_to_modify, replace=False
        )

        for i in modify_indices:
            while True:
                # Generate random values for leaky variables
                L_values = np.random.normal(0.5, 0.1, n_leaky)
                K_value = weights[-1] + sum(weights[j] * L_values[j]
                                            for j in range(n_leaky))

                # Ensure the new K is outside the interval
                if not (intervals[group]['min'] <= K_value <= intervals[group]['max']):
                    df.loc[i, [f'L{j}' for j in range(n_leaky)]] = L_values
                    df.loc[i, 'K'] = K_value
                    break

    # Assign new values to leaky variables
    X_l = df.drop(columns=['K', 'sensitive']).to_numpy()

    if formula:
        print('Formula:')
        print(
            f'K = {weights[-1]} + '.join([f'({weights[i]} * L{i})' for i in range(n_leaky)]))

    if return_weights:
        return X_l, weights

    return X_l


def get_group_name(unique_groups, group):
    """
    Map numerical group identifier to descriptive weather-based group name.

    This function provides human-readable names for sensitive attribute groups
    based on the total number of groups in the dataset.

    Parameters:
        unique_groups (ndarray): Array containing all unique group identifiers in the dataset. The length determines which naming convention to use. Example: [0, 1] or [0, 1, 2, 3].
        group (int): Numerical identifier representing a specific group.

    Returns:
        str: Descriptive group name.

    Note:
        Supported group configurations with weather names:

            - 2 groups: Sunny, Cloudy

            - 3 groups: Sunny, Cloudy, Rainy  

            - 4 groups: Sunny, Cloudy, Rainy, Windy

            - 5 groups: Sunny, Cloudy, Rainy, Windy, Stormy
    """

    if len(unique_groups) == 2:
        return {0: "Sunny", 1: "Cloudy"}.get(group, "")
    elif len(unique_groups) == 3:
        return {0: "Sunny", 1: "Cloudy", 2: "Rainy"}.get(group, "")
    elif len(unique_groups) == 4:
        return {0: "Sunny", 1: "Cloudy", 2: "Rainy", 3: "Windy"}.get(group, "")
    elif len(unique_groups) == 5:
        return {0: "Sunny", 1: "Cloudy", 2: "Rainy", 3: "Windy", 4: "Stormy"}.get(group, "")
    else:
        return ""


def get_class_name(cls):
    """    
    Converts binary classification labels (0, 1) to descriptive class name.

    Parameters:
        cls (int): Numerical class identifier representing a class (0 or 1).

    Returns:
        str: Descriptive name for the class. Returns empty string for invalid inputs.

            - 0 maps to "Negative class"

            - 1 maps to "Positive class"
    """
    return {0: "Negative class", 1: "Positive class"}.get(cls, "")


def get_params_for_certain_equality_type(equality_type, sensitive_group_count):
    """
    Generate group-specific parameters for different fairness equality types.

    This function provides pre-configured parameter sets that simulate different
    fairness scenarios in machine learning. Each equality type addresses specific
    fairness concerns by adjusting dataset generation parameters.

    Parameters:
        equality_type (str): The fairness criterion to simulate. Must be one of:

            - "Equal quality"
                Ensures the classifier performs equally well for all sensitive groups by adjusting `class_sep`.

            - "Demographic parity"
                Ensures equal proportions of positive and negative samples across groups by adjusting `weights`.

            - "Equal opportunity"
                Ensures equal True Positive Rates (TPR) for all sensitive groups by adjusting `weights` and `class_sep`.


            - "Equalized odds"
                Ensures both TPR and False Positive Rates (FPR) are equal across groups by fine-tuning `class_sep`.

        sensitive_group_count (int): Number of sensitive groups in the dataset (2, 3, 4, or 5). Determines which groups will be included and their parameters.

    Returns:
        dict: Dictionary containing parameters for each group. Parameters may include:

            - 'class_sep' : float
                Controls class separability (affects classification difficulty). Higher values = easier classification for that group.

            - 'weights' : list of float
                Controls class distribution [negative_weight, positive_weight]. Affects the proportion of positive vs negative samples.

        Note:
            - `weights` influence the proportion of positive and negative class samples.
            - `class_sep` determines the separability of clusters, affecting accuracy and other metrics.

        Examples:
            >>> params = get_params_for_certain_equality_type("Equal quality", 2)
            >>> print(params)
            {'Sunny': {'class_sep': 1}, 'Cloudy': {'class_sep': 0.6}}

            >>> params = get_params_for_certain_equality_type("Demographic parity", 3)
            >>> print(params)
            {'Sunny': {'weights': [0.7, 0.3]}, 'Cloudy': {'weights': [0.2, 0.8]}, 
            'Rainy': {'weights': [0.4, 0.6]}}
        """

    if equality_type == "Equal quality":
        if sensitive_group_count == 2:
            return {
                "Sunny": {"class_sep": 1},
                "Cloudy": {"class_sep": 0.6}
            }
        elif sensitive_group_count == 3:
            return {
                "Sunny": {"class_sep": 2},
                "Cloudy": {"class_sep": 1},
                "Rainy": {"class_sep": 0.6}
            }
        elif sensitive_group_count == 4:
            return {
                "Sunny": {"class_sep": 2},
                "Cloudy": {"class_sep": 1.5},
                "Rainy": {"class_sep": 1},
                "Windy": {"class_sep": 0.6}
            }
        elif sensitive_group_count == 5:
            return {
                "Sunny": {"class_sep": 2},
                "Cloudy": {"class_sep": 1.5},
                "Rainy": {"class_sep": 1},
                "Windy": {"class_sep": 0.7},
                "Stormy": {"class_sep": 0.5}
            }

    elif equality_type == "Demographic parity":
        if sensitive_group_count == 2:
            return {
                "Sunny": {"weights": [0.7, 0.3]},
                "Cloudy": {"weights": [0.5, 0.5]}
            }
        elif sensitive_group_count == 3:
            return {
                "Sunny": {"weights": [0.7, 0.3]},
                "Cloudy": {"weights": [0.2, 0.8]},
                "Rainy": {"weights": [0.4, 0.6]}
            }
        elif sensitive_group_count == 4:
            return {
                "Sunny": {"weights": [0.7, 0.3]},
                "Cloudy": {"weights": [0.5, 0.5]},
                "Rainy": {"weights": [0.6, 0.4]},
                "Windy": {"weights": [0.4, 0.6]}
            }
        elif sensitive_group_count == 5:
            return {
                "Sunny": {"weights": [0.8, 0.2]},
                "Cloudy": {"weights": [0.6, 0.4]},
                "Rainy": {"weights": [0.5, 0.5]},
                "Windy": {"weights": [0.4, 0.6]},
                "Stormy": {"weights": [0.3, 0.7]}
            }

    elif equality_type == "Equal opportunity":
        if sensitive_group_count == 2:
            return {
                "Sunny": {"weights": [0.2, 0.8], "class_sep": 1.5},
                "Cloudy": {"weights": [0.7, 0.3], "class_sep": 0.7}
            }
        elif sensitive_group_count == 3:
            return {
                "Sunny": {"weights": [0.3, 0.7], "class_sep": 1.2},
                "Cloudy": {"weights": [0.2, 0.8], "class_sep": 1.5},
                "Rainy": {"weights": [0.8, 0.2], "class_sep": 0.6}
            }
        elif sensitive_group_count == 4:
            return {
                "Sunny": {"weights": [0.2, 0.8], "class_sep": 1.5},
                "Cloudy": {"weights": [0.4, 0.6], "class_sep": 1.0},
                "Rainy": {"weights": [0.6, 0.4], "class_sep": 0.8},
                "Windy": {"weights": [0.7, 0.3], "class_sep": 0.6}
            }
        elif sensitive_group_count == 5:
            return {
                "Sunny": {"weights": [0.2, 0.8], "class_sep": 1.5},
                "Cloudy": {"weights": [0.4, 0.6], "class_sep": 1.2},
                "Rainy": {"weights": [0.6, 0.4], "class_sep": 1.0},
                "Windy": {"weights": [0.7, 0.3], "class_sep": 0.8},
                "Stormy": {"weights": [0.8, 0.2], "class_sep": 0.5}
            }

    elif equality_type == "Equalized odds":
        if sensitive_group_count == 2:
            return {
                "Sunny": {"weights": [0.2, 0.8], "class_sep": 1.5},
                "Cloudy": {"weights": [0.7, 0.3], "class_sep": 0.7}
            }
        elif sensitive_group_count == 3:
            return {
                "Sunny": {"weights": [0.3, 0.7], "class_sep": 1.2},
                "Cloudy": {"weights": [0.2, 0.8], "class_sep": 1.5},
                "Rainy": {"weights": [0.8, 0.2], "class_sep": 0.6}
            }
        elif sensitive_group_count == 4:
            return {
                "Sunny": {"weights": [0.2, 0.8], "class_sep": 1.5},
                "Cloudy": {"weights": [0.4, 0.6], "class_sep": 1.0},
                "Rainy": {"weights": [0.6, 0.4], "class_sep": 0.8},
                "Windy": {"weights": [0.7, 0.3], "class_sep": 0.6}
            }
        elif sensitive_group_count == 5:
            return {
                "Sunny": {"weights": [0.2, 0.8], "class_sep": 1.5},
                "Cloudy": {"weights": [0.2, 0.8], "class_sep": 1.5},
                "Rainy": {"weights": [0.6, 0.4], "class_sep": 1.0},
                "Windy": {"weights": [0.5, 0.5], "class_sep": 0.8},
                "Stormy": {"weights": [0.8, 0.2], "class_sep": 0.5}
            }

    else:
        raise ValueError(f"Unsupported equality type: '{equality_type}'. "
                         f"Supported types are: 'Equal quality', 'Demographic parity', "
                         f"'Equal opportunity', 'Equalized odds'")


def get_group_marker(group):
    """
    Map numerical group identifier to matplotlib marker shape.

    Provides distinct visual markers for plotting different groups
    in scatter plots and other visualizations.

    Parameters: 
        group (int): Numerical identifier for the group.

    Returns:
        str: Matplotlib marker string. One of: 'o', 's', '^', 'D', '*', 'v', '<', '>', 'p', 'h', 'H', '+', 'x', '8'.
    """
    markers = ['o', 's', '^', 'D', '*', 'v',
               '<', '>', 'p', 'h', 'H', '+', 'x', '8']
    return markers[int(group) % len(markers)]
