<h1 align="center">
    Make Unfair Classification
</h1>

A Python library for generating biased classification datasets with intentional unfairness patterns. This tool is designed for researchers and practitioners working on fairness in machine learning, allowing them to create controlled datasets with specific bias patterns for testing fairness algorithms. ⚖️🧪

* **Free software:** MIT license
* **Python**: 3.11, 3.12
* **Operating systems**: Windows, Ubuntu, macOS

## ✨ Features
- **Biased Dataset Generation**: Create classification datasets with intentional bias across sensitive groups. 🗃️
- **Fairness Evaluation**: Built-in tools for evaluating model fairness across different groups. ⚖️
- **Visualization**: Visualization capabilities for understanding bias patterns and fairness metrics. 📈
- **Flexible Configuration**: Support for various equality types (demographic parity, equal opportunity, etc.). ⚙️
- **Leaky Features**: Generate features that leak sensitive information to simulate real-world bias. 🔓

## ⚖️ Supported Equality Types
The library supports generating datasets that violate specific fairness criteria:

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