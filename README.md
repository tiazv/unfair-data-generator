<h1 align="center">
    Make Unfair Classification
</h1>

<p align="center">
    <a href="#-about">📋 About</a> •
    <a href="#-installation">📦 Installation</a> •
    <a href="#-usage">🚀 Usage</a> •
    <a href="#️-supported-equality-types">⚖️ Supported Equality Types</a> •
    <a href="#-license">📜 License</a>
</p>

## 📋 About

**Make Unfair Classification** is a Python library designed for generating biased classification datasets with intentional unfairness patterns. This tool extends scikit-learn's `make_classification` function to include sensitive group information and fairness constraints, allowing users to create controlled datasets with specific bias patterns for testing and developing fairness algorithms. ⚖️🧪

**Make Unfair Classification** supports various fairness criteria violations and provides comprehensive tools for visualization and evaluation, making it an essential tool for fairness research and education. 💡

* **Free software:** MIT license
* **Python**: 3.11, 3.12
* **Operating systems**: Windows, Ubuntu, macOS

## ✨ Features
- **Biased Dataset Generation**: Create classification datasets with intentional bias 
across sensitive groups. 🗃️
- **Fairness Evaluation**: Built-in tools for evaluating model fairness across different 
groups. ⚖️
- **Visualization**: Visualization capabilities for understanding bias patterns and 
fairness metrics. 📈
- **Flexible Configuration**: Support for various equality types (demographic parity, 
equal opportunity, equal opportunity, equalized odds). ⚙️
- **Leaky Features**: Generate features that leak sensitive information to simulate 
real-world bias. 🔓
- **🌦️ Multiple Groups**: Support for 2-5 sensitive groups with intuitive weather-based naming
- **🎯 Scikit-learn Compatible**: Extends familiar scikit-learn patterns and interfaces

## 📦 Installation


## 🚀 Usage

The following example demonstrates how to generate a biased dataset and evaluate fairness using `make-unfair-classification`. More examples can be found in the [examples](./examples) directory.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from make_unfair_classification.unfair_classification import make_unfair_classification
from make_unfair_classification.util.helpers import get_params_for_certain_equality_type
from make_unfair_classification.util.model_trainer import train_and_evaluate_model_with_classifier
from make_unfair_classification.util.visualizer import (
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
```

## ⚖️ Supported Equality Types

The library supports generating datasets that systematically violate specific fairness criteria. Each type creates different bias patterns:

- **Equal quality**   
Different classification performance across groups.
- **Demographic parity**  
Unequal positive prediction rates across groups.
- **Equal opportunity**  
Unequal true positive rates across groups.
- **Equalized odds**  
Unequal true positive and false positive rates across groups.

## 📜 License
This package is distributed under the MIT License. This license can be found online at <http://www.opensource.org/licenses/MIT>.

## Disclaimer
This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!