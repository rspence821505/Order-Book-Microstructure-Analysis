![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

# Order-Book-Microstructure-Analysis

<!-- # Limit Order Book Feature Pipeline & Tree-Based Prediction -->

> End-to-end pipeline for engineering microstructure features, detecting market regimes, and predicting mid-price direction using tree-based models.

## 🎯 Project Overview

This project demonstrates:

- Engineering 45+ microstructure features from limit order book data
- Hawkes process modeling for order flow dynamics
- Multi-method regime detection (HMM + Hawkes)
- Tree-based prediction models (Decision Trees, Random Forests, Gradient Boosting)
- Comprehensive model interpretability (SHAP, permutation importance, partial dependence)
- Regime-conditional modeling and economic validation

**Key Results:**

<!-- - 68% directional accuracy (vs. 53% baseline)
- 22% Sharpe ratio improvement with regime-aware trading
- Sub-400μs inference latency for production deployment -->

## 📊 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/lob-feature-pipeline-prediction.git
cd lob-feature-pipeline-prediction
pip install -r requirements.txt
pip install -e .  # Install src/ as package
```
