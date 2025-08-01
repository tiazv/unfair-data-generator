Make Unfair Classification
==========================

A Python library for generating biased classification datasets with intentional unfairness patterns. This tool is designed for researchers and practitioners working on fairness in machine learning, allowing them to create controlled datasets with specific bias patterns for testing fairness algorithms. ⚖️🧪

* **Free software:** MIT license
* **GitHub**: https://github.com/tiazv/Make-Unfair-Classification
* **Python**: 3.11, 3.12
* **Operating systems**: Windows, Ubuntu, macOS

Features
--------

- **Biased Dataset Generation**: Create classification datasets with intentional bias across sensitive groups. 🗃️
- **Fairness Evaluation**: Built-in tools for evaluating model fairness across different groups. ⚖️
- **Visualization**: Visualization capabilities for understanding bias patterns and fairness metrics. 📈
- **Flexible Configuration**: Support for various equality types (demographic parity, equal opportunity, etc.). ⚙️
- **Leaky Features**: Generate features that leak sensitive information to simulate real-world bias. 🔓

Supported Equality Types
------------------------

The library supports generating datasets that violate specific fairness criteria:

- **"Equal quality"**: Different classification performance across groups.
- **"Demographic parity"**: Unequal positive prediction rates across groups.
- **"Equal opportunity"**: Unequal true positive rates across groups.
- **"Equalized odds"**: Unequal true positive and false positive rates across groups.

Documentation
-------------

The documentation is organised into the following sections:

* :ref:`make_unfair_classification`
* :ref:`about`

.. _dev:

.. toctree::
   :maxdepth: 1
   :caption: Developer documentation

   dev/installation
   dev/documentation

.. _make_unfair_classification:

.. toctree::
   :maxdepth: 2
   :caption: Make Unfair Classification

   make_unfair_classification/index

.. _about:

.. toctree::
   :maxdepth: 1
   :caption: About

   about/license
   about/code_of_conduct