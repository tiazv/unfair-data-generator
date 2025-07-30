import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

try:
    from .helpers import get_group_name, get_class_name, get_group_marker
except ImportError:
    from helpers import get_group_name, get_class_name, get_group_marker

"""
Visualization utilities for fairness analysis in machine learning.

This module provides methods for generating and visualizing various fairness 
metrics, accuracy metrics, and group-specific cluster visualizations for 
datasets.

Methods:
---------
1. visualize_TPR_FPR_metrics(metrics, title)
    Creates bar charts to visualize True Positive Rate (TPR) and False Positive Rate (FPR) for different groups.

2. visualize_accuracy(metrics, title)
    Creates bar charts to visualize accuracy for different groups.

3. visualize_groups_separately(X, y, title="Group-specific visualization")
    Generates scatter plots for individual groups, showing data points for each class.

4. visualize_group_classes : function
    Generates scatter plots for all groups and their respective centroids.
"""

def visualize_TPR_FPR_metrics(metrics, title):
    """
    Create bar charts to visualize True Positive Rate (TPR) and False Positive Rate (FPR) for different groups.

    Parameters:
    ----------
    metrics : dict
        Dictionary containing fairness metrics for each group. Each group should 
        have 'True Positive Rate (TPR)' and 'False Positive Rate (FPR)' keys.

    title : str
        Title for the plot.

    Returns:
    -------
    matplotlib.pyplot
        The matplotlib pyplot object containing the group cluster visualization.
    """
    groups = list(metrics.keys())
    tpr = [metrics[group]["True Positive Rate (TPR)"] for group in groups]
    fpr = [metrics[group]["False Positive Rate (FPR)"] for group in groups]

    x = np.arange(len(groups))

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, tpr, width=0.4, label="True Positive Rate (TPR)")
    plt.bar(x + 0.2, fpr, width=0.4, label="False Positive Rate (FPR)")
    plt.xticks(x, groups)
    plt.ylabel("Rate")
    plt.title(f"Fairness Metrics - {title}")
    plt.legend()

    return plt


def visualize_accuracy(metrics, title):
    """
    Create bar charts to visualize accuracy for different groups.

    Parameters:
    ----------
    metrics : dict
        Dictionary containing accuracy metrics for each group.

    title : str
        Title for the plot.

    Returns:
    -------
    matplotlib.pyplot
        The matplotlib pyplot object containing the group cluster visualization.
    """
    groups = list(metrics.keys())
    apr = [metrics[group]["Accuracy"] for group in groups]

    x = np.arange(len(groups))

    plt.figure(figsize=(10, 6))
    plt.bar(x, apr, label="Accuracy")
    plt.xticks(x, groups)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Metrics - {title}")

    return plt


def visualize_groups_separately(X, y, Z, feature1=None, feature2=None, feature1_name=None, feature2_name=None, title="Group-specific visualization"):
    """
    Generate scatter plots for individual groups, showing data points for each class.

    Parameters:
    ----------
    X : ndarray
        Feature matrix. When using default features, the first two columns are used.

    y : ndarray
        Target class labels.

    Z : ndarray
        Sensitive group labels for each sample.

    feature1 : ndarray, optional
        First feature for visualization. If None, uses X[:, 0].

    feature2 : ndarray, optional
        Second feature for visualization. If None, uses X[:, 1].

    feature1_name : str, optional
        Name for the first feature axis label. Defaults to "Feature 1".

    feature2_name : str, optional
        Name for the second feature axis label. Defaults to "Feature 2".

    title : str, default="Group-specific visualization"
        Base title for the plots.

    Returns:
    -------
    figures : dict
        Dictionary containing matplotlib figure objects for each group.
    """
    
    if feature1 is None and feature2 is None:
        # Default: use first two columns of X
        f1 = X[:, 0]
        f2 = X[:, 1]
        
        f1_name = feature1_name if feature1_name is not None else "Feature 1"
        f2_name = feature2_name if feature2_name is not None else "Feature 2"
        
    elif feature1 is not None and feature2 is not None:
        # Custom features
        f1 = feature1
        f2 = feature2
        
        f1_name = feature1_name if feature1_name is not None else "Feature 1"
        f2_name = feature2_name if feature2_name is not None else "Feature 2"
        
        if np.array_equal(f1, f2):
            raise ValueError("Feature 1 and Feature 2 must be different. Please provide different feature columns.")
    else:
        raise ValueError("Feature 1 and Feature 2 must be provided together, or both should be None for default behavior.")

    sensitive_group_labels = Z
    unique_groups = np.unique(sensitive_group_labels)

    # Calculate global X and Y axis limits using the selected features
    x_min, x_max = f1.min() - 1, f1.max() + 1
    y_min, y_max = f2.min() - 1, f2.max() + 1

    figures = {}

    for group in unique_groups:
        plt.subplots(figsize=(8, 6))

        # Mask for the current group
        group_mask = sensitive_group_labels == group

        group_name = get_group_name(unique_groups, group)

        # Scatter plot for each class within the group
        for cls in np.unique(y):
            class_name = get_class_name(cls)

            class_mask = y == cls
            combined_mask = group_mask & class_mask
            plt.scatter(f1[combined_mask], f2[combined_mask], 
                      label=f"{class_name}", s=60, alpha=0.7)

        # Apply global limits for X and Y axes
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.title(f"{title} - {group_name}")
        plt.xlabel(f1_name)
        plt.ylabel(f2_name)
        plt.legend()

    return figures

def visualize_group_classes(X, y, Z, centroids, feature1=None, feature2=None, feature1_name=None, feature2_name=None, title="Group cluster visualization"):
    """
    Visualize data points for group-class combinations and centroids.

    Parameters:
    ----------
    X : ndarray
        Feature matrix. When using default features, the first two columns are used.

    y : ndarray
        Target class labels.

    centroids : dict
        Dictionary of group-specific centroids.

    Z : ndarray
        Sensitive group labels for each sample.

    feature1 : ndarray, optional
        First feature for visualization. If None, uses X[:, 0].

    feature2 : ndarray, optional
        Second feature for visualization. If None, uses X[:, 1].

    feature1_name : str, optional
        Name for the x-axis label. Defaults to "Feature 1" or "Custom Feature 1".

    feature2_name : str, optional
        Name for the y-axis label. Defaults to "Feature 2" or "Custom Feature 2".

    title : str, default="Group cluster visualization"
        Title for the plot.

    Returns:
    -------
    matplotlib.pyplot
        The matplotlib pyplot object containing the group cluster visualization.
    """
    
    if feature1 is None and feature2 is None:
        # Default: use first two columns of X and centroids
        f1 = X[:, 0]
        f2 = X[:, 1]
        
        f1_name = feature1_name if feature1_name is not None else "Feature 1"
        f2_name = feature2_name if feature2_name is not None else "Feature 2"
        
        # Extract centroids for first and second dimensions
        c1 = {group: [centroid[0] for centroid in centroid_list] for group, centroid_list in centroids.items()}
        c2 = {group: [centroid[1] for centroid in centroid_list] for group, centroid_list in centroids.items()}
        
    elif feature1 is not None and feature2 is not None:
        # Custom features
        f1 = feature1
        f2 = feature2
        
        f1_name = feature1_name if feature1_name is not None else "Feature 1"
        f2_name = feature2_name if feature2_name is not None else "Feature 2"
        
        # Validate that features are different
        if np.array_equal(f1, f2):
            raise ValueError("Feature 1 and Feature 2 must be different. Please provide different feature columns.")
        
        # Auto-detect which columns the features correspond to
        feature1_idx = None
        feature2_idx = None
        
        # Check if features match any columns in X
        max_col = X.shape[1]
        
        for idx in range(max_col):
            if np.array_equal(f1, X[:, idx]):
                feature1_idx = idx
            if np.array_equal(f2, X[:, idx]):
                feature2_idx = idx
        
        # Extract centroid dimensions automatically with validation
        if feature1_idx is not None:
            # Check if centroids have enough dimensions
            sample_centroid = next(iter(centroids.values()))[0]
            if feature1_idx >= len(sample_centroid):
                raise ValueError(f"Feature1 corresponds to column {feature1_idx}, but centroids only have {len(sample_centroid)} dimensions (0-{len(sample_centroid)-1}).")
            c1 = {group: [centroid[feature1_idx] for centroid in centroid_list] for group, centroid_list in centroids.items()}
        else:
            raise ValueError("Could not auto-detect centroids for feature1. Feature1 doesn't match any column of X.")
            
        if feature2_idx is not None:
            # Check if centroids have enough dimensions
            sample_centroid = next(iter(centroids.values()))[0]
            if feature2_idx >= len(sample_centroid):
                raise ValueError(f"Feature2 corresponds to column {feature2_idx}, but centroids only have {len(sample_centroid)} dimensions (0-{len(sample_centroid)-1}).")
            c2 = {group: [centroid[feature2_idx] for centroid in centroid_list] for group, centroid_list in centroids.items()}
        else:
            raise ValueError("Could not auto-detect centroids for feature2. Feature2 doesn't match any column of X.")
    else:
        raise ValueError("Feature 1 and Feature 2 must be provided together, or both should be None for default behavior.")
    
    group_labels = Z
    unique_groups = np.unique(group_labels)
    unique_classes = np.unique(y)

    group_class_colors = {}  # Dynamic dictionary for colors
    colormap = cm.get_cmap("tab10", len(unique_groups) * len(unique_classes))  # Get a colormap

    color_index = 0  # Track color indices
    for group in unique_groups:
        for cls in unique_classes:
            group_class_colors[(group, cls)] = colormap(color_index)
            color_index += 1

    # Plot data points for each group-class combination
    plt.subplots(figsize=(10, 8))
    for (group, cls), color in group_class_colors.items():
        mask = (group_labels == group) & (y == cls)

        class_name = get_class_name(cls)
        group_name = get_group_name(unique_groups, group)
        group_marker = get_group_marker(group)

        plt.scatter(f1[mask], f2[mask], c=[color], 
                  label=f"{class_name} - {group_name}", 
                  s=60, alpha=0.7, marker=group_marker)
    
    # Create mapping between group IDs and centroid keys
    sorted_groups = sorted(unique_groups)
    sorted_centroid_keys = sorted(c1.keys())
    
    group_to_centroid_key = {}
    for i, group in enumerate(sorted_groups):
        if i < len(sorted_centroid_keys):
            group_to_centroid_key[group] = sorted_centroid_keys[i]
    
    for group in unique_groups:
        # Use the mapping to get the correct centroid key
        if group in group_to_centroid_key:
            centroid_key = group_to_centroid_key[group]
            
            if centroid_key in c1 and centroid_key in c2:
                group_name = get_group_name(unique_groups, group)
                plt.scatter(
                    c1[centroid_key],
                    c2[centroid_key],
                    color="black", marker="x", s=100,
                    label=f"{centroid_key} Centroids",
                )

    # Finalize the plot
    marker_legend = []
    for i, group in enumerate(unique_groups):
        group_name = get_group_name(unique_groups, group)
        group_marker = get_group_marker(group)
        marker_legend.append(f"{group_name}({group_marker})")
    
    plt.title(title)
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.legend()

    return plt