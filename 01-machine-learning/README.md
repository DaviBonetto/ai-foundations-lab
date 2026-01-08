# 🤖 Machine Learning Foundations

A comprehensive exploration of fundamental machine learning algorithms implemented from scratch. This section focuses on building intuition through first-principles implementations, covering both supervised and unsupervised learning paradigms.

## 📋 Overview

Machine Learning forms the foundation of modern AI systems. Rather than using black-box libraries, every algorithm here is implemented from scratch using NumPy, allowing for deep understanding of the mathematical principles and computational mechanics behind each method.

## 📂 Contents

### Supervised Learning

#### 📊 **Linear Regression from Scratch**
*File: `linear-regression-from-scratch.ipynb`*

Complete implementation of linear regression using gradient descent, exploring:
- Cost function (MSE) derivation and optimization
- Gradient descent algorithm with learning rate tuning
- Feature normalization and its impact on convergence
- Analytical solution via Normal Equation
- Visualization of decision boundaries and error surfaces

**Key Concepts:** Hypothesis function, cost minimization, convex optimization

---

#### 🎯 **Logistic Regression Implementation**
*File: `logistic-regression-implementation.ipynb`*

Binary classification using logistic regression, covering:
- Sigmoid activation function and probabilistic interpretation
- Cross-entropy loss function derivation
- Gradient descent for non-linear optimization
- Decision boundaries in feature space
- Extension to multiclass classification (one-vs-all)

**Key Concepts:** Maximum likelihood estimation, log-odds, classification metrics

---

#### 🔍 **Support Vector Machines with Kernels**
*File: `svm-kernels.ipynb`*

Exploring SVMs for both linear and non-linear classification:
- Hard-margin and soft-margin formulations
- Kernel trick: polynomial, RBF (Gaussian), and custom kernels
- Support vectors identification and interpretation
- Hyperparameter tuning (C, gamma)
- Comparison of kernel performance on different datasets

**Key Concepts:** Maximum margin, kernel methods, dual problem, slack variables

---

### Unsupervised Learning

#### 🎨 **K-Means Clustering**
*File: `k-means-clustering.ipynb`*

Centroid-based clustering algorithm implementation:
- Random initialization and centroid update rules
- Assignment step and update step mechanics
- Elbow method for optimal k selection
- Convergence criteria and iteration limits
- Visualization of cluster formation over iterations

**Key Concepts:** Distance metrics (Euclidean), inertia, cluster cohesion

---

#### 📉 **PCA - Dimensionality Reduction**
*File: `pca-dimensionality-reduction.ipynb`*

Principal Component Analysis for feature extraction and visualization:
- Covariance matrix computation and eigendecomposition
- Principal components as directions of maximum variance
- Dimensionality reduction while preserving information
- Reconstruction error analysis
- Applications: data compression and visualization

**Key Concepts:** Eigenvectors, variance explained, orthogonal projections

---

## 🎯 Learning Objectives

By working through these implementations, you will understand:

- ✅ The mathematical foundations of ML algorithms
- ✅ How optimization (gradient descent) works under the hood
- ✅ Trade-offs between model complexity and generalization
- ✅ When to use supervised vs. unsupervised methods
- ✅ How to debug and improve ML models from first principles

## 🛠️ Technologies Used

- **Python 3.8+**
- **NumPy** - Core numerical computations
- **Matplotlib** - Visualizations and plots
- **Scikit-learn** - Dataset loading and comparison benchmarks
- **Jupyter Notebooks** - Interactive development

## 🚀 Getting Started
```bash
# Navigate to the machine learning directory
cd 01-machine-learning

# Install required packages
pip install numpy matplotlib scikit-learn jupyter

# Launch Jupyter Notebook
jupyter notebook
```

Then open any `.ipynb` file to explore the implementations.

## 📊 Repository Structure
```
01-machine-learning/
├── supervised-learning/
│   ├── linear-regression-from-scratch.ipynb
│   ├── logistic-regression-implementation.ipynb
│   └── svm-kernels.ipynb
├── unsupervised-learning/
│   ├── k-means-clustering.ipynb
│   └── pca-dimensionality-reduction.ipynb
└── README.md (you are here)
```

## 📚 Learning Resources

These implementations are inspired by and aligned with:
- **Stanford CS229: Machine Learning** (Andrew Ng)
- **Pattern Recognition and Machine Learning** (Christopher Bishop)
- **The Elements of Statistical Learning** (Hastie, Tibshirani, Friedman)

## 🔗 Navigation

[← Back to Main Repository](../README.md) | [Next: Deep Learning →](../02-deep-learning/README.md)

---

**Part of the [AI Foundations Lab](../README.md) project - A self-directed journey through ML, DL, NLP, and RL.**
