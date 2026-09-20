# Machine Learning Experiments: Classification, Dimensionality Reduction and Neural Networks

This repository contains two Jupyter notebooks exploring different machine learning techniques across **classification, regression, clustering, and image classification**.

The main goal is to compare different machine learning models and investigate how **dimensionality reduction techniques**, particularly **Principal Component Analysis (PCA)** and **Autoencoders**, affect model performance.

## Repository Structure

```text
.
├── First.ipynb
└── Second.ipynb
```

`First.ipynb` focuses on **phishing website detection using traditional machine learning classifiers**.

`Second.ipynb` investigates **dimensionality reduction with PCA and Autoencoders** across three different machine learning problems:

```text
Regression      → Air Quality prediction
Clustering      → Travel Review clustering
Classification  → Formula 1 car image classification
```

---

# 1. Phishing Website Classification — `First.ipynb`

The first notebook investigates whether machine learning models can distinguish between **phishing and legitimate websites**.

The dataset is retrieved from the **UCI Machine Learning Repository — Phishing Websites dataset**.

Each website is represented through features describing properties of the URL and webpage, such as URL length, use of IP addresses, subdomains, SSL information, domain age, redirects, and other characteristics commonly associated with phishing websites.

## Workflow

The notebook follows the pipeline:

```text
Phishing Website Dataset
        ↓
Train/Test Split
        ↓
Standardization
        ↓
PCA
        ↓
Machine Learning Model
        ↓
Hyperparameter Tuning
        ↓
Evaluation
```

The dataset is divided into:

```text
80% Training
20% Testing
```

A fixed `random_state=42` is used to make the split reproducible.

---

## Preprocessing

Each model is placed inside a Scikit-learn `Pipeline`.

The pipeline first applies:

### StandardScaler

The features are standardized so that they approximately have:

```text
mean = 0
standard deviation = 1
```

This is particularly important for algorithms such as Logistic Regression and Support Vector Machines.

### PCA

Principal Component Analysis is then applied using:

```python
PCA(n_components=0.95)
```

Instead of specifying a fixed number of dimensions, PCA retains enough principal components to preserve **95% of the variance in the dataset**.

The objective is to reduce dimensionality while preserving most of the information contained in the original features.

---

# Models Tested

Six classification algorithms are compared:

```text
Naive Bayes
Logistic Regression
Decision Tree
Random Forest
SVM with RBF kernel
SVM with Linear kernel
```

Each model has its own set of hyperparameters.

For example, Random Forest evaluates different numbers of trees and maximum tree depths, while the SVM models explore different values of `C` and `gamma`.

---

# Hyperparameter Tuning

Hyperparameter optimization is performed using:

```python
GridSearchCV
```

with **5-fold cross-validation**.

For each hyperparameter configuration, the training data is divided into five parts.

The model is trained on four parts and validated on the remaining part. This process is repeated five times.

The main metric used to select the best configuration is:

```text
F1 Score
```

F1 is particularly useful for classification because it combines **precision and recall**:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The notebook also records how long hyperparameter tuning takes for each model.

---

# Evaluation

After the best version of every model has been selected, it is evaluated on the previously unseen test set.

The notebook calculates:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
```

ROC curves and learning curves are also generated.

The learning curves compare training and validation performance as progressively more training data is used. They can help identify whether a model is suffering from **overfitting or underfitting**.

---

# Phishing Classification Results

The results stored in the notebook are approximately:

| Model               |  Accuracy |        F1 |   ROC-AUC |
| ------------------- | --------: | --------: | --------: |
| Naive Bayes         |     0.897 |     0.910 |     0.960 |
| Logistic Regression |     0.923 |     0.933 |     0.977 |
| Decision Tree       |     0.945 |     0.952 |     0.958 |
| Random Forest       |     0.958 |     0.963 | **0.991** |
| SVM RBF             | **0.962** | **0.967** |     0.991 |
| SVM Linear          |     0.926 |     0.936 |     0.977 |

In this experiment, the **RBF SVM obtains the highest test accuracy and F1 score**, while Random Forest achieves the highest ROC-AUC by a small margin.

The experiment therefore also demonstrates why comparing several metrics is important: the "best" model can depend on which aspect of classification performance is being measured.

---

# 2. Dimensionality Reduction Experiments — `Second.ipynb`

The second notebook investigates a different question:

> How does reducing the dimensionality of the data affect machine learning performance?

Three representations of the same data are compared:

```text
Original features
        vs
PCA-reduced features
        vs
Autoencoder-reduced features
```

The experiment is repeated on three different types of machine learning problems:

```text
Regression
Clustering
Image Classification
```

This allows PCA and Autoencoders to be compared under very different conditions.

---

# Experiment 1 — Air Quality Regression

The first experiment uses the **UCI Air Quality dataset**.

The task is to predict:

```text
CO(GT)
```

which represents carbon monoxide concentration.

## Data Cleaning

The dataset uses `-200` to represent missing measurements.

These values are replaced with:

```python
NaN
```

Rows containing too many missing features are removed.

The remaining missing values are replaced using the mean of each feature.

`Date`, `Time`, and the target variable `CO(GT)` are removed from the predictor matrix.

The features are then standardized using `StandardScaler`.

---

## Data Split

The data is divided approximately into:

```text
70% training
15% validation
15% testing
```

Three different versions of the dataset are then created.

### Original

The standardized features are used directly.

### PCA

PCA compresses the data to:

```text
10 dimensions
```

### Autoencoder

A neural network Autoencoder also compresses the features into a:

```text
10-dimensional latent representation
```

The Autoencoder learns to reproduce its own input:

```text
Input
  ↓
32 neurons
  ↓
10 neurons        ← compressed representation
  ↓
32 neurons
  ↓
Reconstructed input
```

After training, only the **encoder** portion is used to transform the original data.

---

# Regression Neural Network

A Feedforward Neural Network is then trained separately on:

```text
Original features
PCA features
Autoencoder features
```

The architecture and training parameters are automatically tuned using `RandomizedSearchCV`.

The search explores parameters including:

```text
Number of hidden layers
Number of neurons
Dropout
Learning rate
Batch size
Number of epochs
```

Three-fold cross-validation is used during the search.

Early stopping is also used to stop training when validation performance no longer improves.

---

## Regression Metrics

The models are evaluated using:

```text
MSE   — Mean Squared Error
RMSE  — Root Mean Squared Error
MAE   — Mean Absolute Error
R²    — Coefficient of Determination
```

The recorded results are:

| Representation |      RMSE |       MAE |        R² |
| -------------- | --------: | --------: | --------: |
| Original       | **0.374** |     0.234 | **0.930** |
| PCA            |     0.382 | **0.233** |     0.927 |
| Autoencoder    |     0.404 |     0.252 |     0.919 |

The dimensionality-reduced representations therefore do **not improve predictive performance in this experiment**.

The original representation produces the highest R², although PCA remains relatively close.

---

# Experiment 2 — Travel Review Clustering

The second experiment uses the **UCI Travel Reviews dataset**.

This is an **unsupervised learning** problem.

Instead of predicting an output label, the objective is to discover groups of similar reviews.

Missing values are replaced with feature means and the data is standardized.

Again, three representations are compared:

```text
Original → 10 dimensions

PCA → 5 dimensions

Autoencoder → 5 dimensions
```

---

# K-Means Clustering

K-Means is used to divide the observations into clusters.

The notebook tests:

```text
k = 2, 3, 4, 5, 6, 7, 8
```

Rather than arbitrarily selecting the number of clusters, each value of `k` is evaluated.

The best `k` is selected primarily according to the **Silhouette Score**.

Three internal clustering metrics are calculated:

### Silhouette Score

Measures how well samples fit their own cluster compared with neighbouring clusters.

```text
Higher = better
```

### Davies-Bouldin Index

Measures similarity between clusters.

```text
Lower = better
```

### Calinski-Harabasz Score

Compares separation between clusters with variation inside clusters.

```text
Higher = better
```

---

## Clustering Results

The selected models produce approximately:

| Representation | Best k | Silhouette | Davies-Bouldin | Calinski-Harabasz |
| -------------- | -----: | ---------: | -------------: | ----------------: |
| Original       |      2 |      0.213 |          1.796 |            255.93 |
| PCA            |      2 |      0.272 |          1.468 |            368.44 |
| Autoencoder    |      3 |  **0.346** |      **1.044** |        **383.82** |

Unlike the regression experiment, dimensionality reduction improves the internal clustering metrics.

The **Autoencoder representation produces the strongest clustering scores in this run**.

This suggests that the nonlinear representation learned by the Autoencoder may make the groups in this dataset easier for K-Means to separate.

---

# Experiment 3 — Formula 1 Car Image Classification

The final experiment uses a Formula 1 car image dataset downloaded through KaggleHub.

The dataset contains ten classes:

```text
Alfa Romeo
BWT
Ferrari
Haas
McLaren
Mercedes
Red Bull
Renault
Toro Rosso
Williams
```

Images are resized to:

```text
64 × 64 × 3
```

and their pixel values are normalized to the range:

```text
[0, 1]
```

The training set contains 6,616 images.

The supplied validation dataset is divided into validation and test sets.

---

# Image Dimensionality Reduction

Images contain significantly more dimensions than the previous tabular datasets.

A 64 × 64 RGB image contains:

```text
64 × 64 × 3 = 12,288 values
```

The notebook therefore compares the original images against two compressed representations containing only:

```text
256 features
```

---

## PCA Image Representation

Each image is first flattened into a vector.

PCA compresses the original 12,288 values into:

```text
256 principal components
```

The 256 values are reshaped into:

```text
16 × 16 × 1
```

so they can be provided to the CNN.

---

## Autoencoder Image Representation

A dense Autoencoder is also trained to compress each image:

```text
12,288 input values
        ↓
256 hidden units
        ↓
256-dimensional latent representation
        ↓
256 hidden units
        ↓
12,288 reconstructed values
```

The learned latent vector is also reshaped into:

```text
16 × 16 × 1
```

---

# Convolutional Neural Network

Three CNNs are trained:

```text
CNN using original images

CNN using PCA representation

CNN using Autoencoder representation
```

The basic CNN architecture contains three convolutional stages:

```text
Input
  ↓
Conv2D
  ↓
MaxPooling
  ↓
Conv2D
  ↓
MaxPooling
  ↓
Conv2D
  ↓
MaxPooling
  ↓
Flatten
  ↓
Dense
  ↓
Softmax
```

Different combinations of filter counts, dense-layer sizes, dropout, and learning rates are tested.

The configuration producing the highest validation accuracy is retained.

---

# Image Classification Results

The recorded test results are:

| Representation  |  Accuracy |  Macro F1 |
| --------------- | --------: | --------: |
| Original images | **0.599** | **0.594** |
| PCA             |     0.249 |     0.223 |
| Autoencoder     |     0.261 |     0.228 |

The original image representation performs substantially better than both dimensionality-reduced representations.

This result is important.

CNNs are designed to exploit **spatial relationships between neighbouring pixels**. PCA and the dense Autoencoder transform an image into a vector of abstract features.

Reshaping those 256 features into a `16 × 16` array makes them compatible with a CNN, but it does **not restore the original spatial structure of the image**.

For example, two neighbouring values in the resulting `16 × 16` representation are not necessarily related to neighbouring parts of the Formula 1 car.

This is a likely reason why the CNN performs much worse on the compressed representations.

A convolutional Autoencoder would provide a more natural extension for this experiment because it can preserve more spatial information.

---

# Main Findings

Across the three experiments, dimensionality reduction has different effects depending on the task.

```text
Air Quality Regression
Original ≈ PCA > Autoencoder

Travel Review Clustering
Autoencoder > PCA > Original

F1 Image Classification
Original >> Autoencoder ≈ PCA
```

The experiments demonstrate that dimensionality reduction should not automatically be expected to improve machine learning performance.

PCA and Autoencoders can reduce computational dimensionality and sometimes reveal useful representations, but information can also be lost during compression.

The effectiveness of a reduced representation therefore depends on both the **dataset and the downstream machine learning task**.

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* SciKeras
* UCI Machine Learning Repository
* KaggleHub

---

# Important Implementation Notes

The notebooks are experimental and there are several aspects that could be improved before treating the results as a rigorous benchmark.

In the Air Quality experiment, missing-value imputation and `StandardScaler` are currently applied **before the train/test split**. This means information from the future test set contributes slightly to preprocessing and creates data leakage. A stronger implementation would split the data first and fit the imputer and scaler using the training data only.

The phishing notebook also produces Scikit-learn warnings because the target is stored as a one-column DataFrame rather than a one-dimensional array. Converting it using something such as:

```python
y = y.values.ravel()
```

would remove this warning.

Finally, applying PCA before Decision Trees and Random Forests is not inherently necessary. Tree-based algorithms do not require standardized features and can often benefit from retaining the original feature representation. Comparing these models both **with and without PCA** would make the experiment more informative.

---

# Possible Future Improvements

Potential extensions include evaluating models without dimensionality reduction, using stratified dataset splits for classification, adding statistical comparisons between models, using Convolutional Autoencoders for image compression, visualizing learned latent spaces with t-SNE or UMAP, adding model explainability, and investigating how the number of retained PCA or Autoencoder dimensions affects accuracy and training time.

---

## Purpose

These notebooks were created to explore both traditional and neural machine learning techniques and, in particular, to study the trade-off between **data compression, computational cost, and predictive performance**.

Rather than assuming that dimensionality reduction is always beneficial, the experiments demonstrate that its usefulness depends strongly on the structure of the data and the machine learning problem being solved.
