# Clinical Acquisition Bias Audit

Official implementation accompanying the manuscript:

**Adversarial Validation Reveals Diagnostic Workflow Leakage in PCOS Machine Learning Models**

---

## Overview

Retrospective clinical datasets often contain hidden acquisition-related information that can substantially inflate machine learning performance. Diagnostic workflow, informative missingness, duplicated schema, and measurement availability may encode disease labels independently of the underlying biological signal.

This repository implements a reproducible framework for detecting and quantifying these sources of bias before interpreting diagnostic model performance.

The proposed audit framework separates information originating from:

- raw database schema
- schema harmonization
- informative missingness
- measured clinical values
- acquisition-balanced feature subsets
- demographic effects
- genuine residual biological signal

The framework is intended for retrospective clinical machine learning studies and is not limited to PCOS.

---

## Features

- Schema audit
- Clinical concept harmonization
- Missingness analysis
- Adversarial validation
- Layered acquisition-bias framework
- Ascertainment-balanced feature selection
- Bootstrap inference
- Permutation testing
- Calibration analysis
- Semi-synthetic validation
- Publication-ready figures and tables

---


---

## Methodological framework

The audit consists of six analytical layers:

1. Raw-schema missingness
2. Harmonized missingness
3. Raw values
4. Values + missingness indicators
5. Ascertainment-balanced models
6. Ascertainment-balanced models excluding age

Performance is evaluated using:

- ROC-AUC
- Average Precision
- Balanced Accuracy
- Matthews Correlation Coefficient
- Brier Score
- Log Loss
- Calibration
- Bootstrap confidence intervals
- Label permutation testing

---

## Citation

If you use this repository, please cite:

Kataryńczuk K, Stachowiak A, Piorkowska N, Ostromęcki A, Franik G, Bizoń A.

*Adversarial Validation Reveals Diagnostic Workflow Leakage in PCOS Machine Learning Models.*

Artificial Intelligence in Medicine (under review).

---

## License

MIT License

---

## Contact

Natalia Piórkowska

Faculty of Information and Communication Technology

Wroclaw University of Science and Technology

Email:
natalia.piorkowska@pwr.edu.pl

