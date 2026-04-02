[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/U0WGAoAy)
# AI Lab — Naive Bayes and K-Nearest Neighbors

This assignment covers two basic supervised learning methods:

1. **Naive Bayes** for spam / non-spam email classification
2. **K-Nearest Neighbors (KNN)** for classification on the Iris dataset

---

## Repository Structure

```text
AIstats_lab.py
test_AIstats_lab.py
README.md
```

---

## Q1 — Naive Bayes Spam Classifier

Implement a **Naive Bayes classifier** using **simple Maximum Likelihood Estimation (MLE)**.

### Dataset

Use this dataset inside your function:

```python
texts = [
    "win money now",
    "limited offer win cash",
    "cheap meds available",
    "win big prize now",
    "exclusive offer buy now",
    "cheap pills buy cheap meds",
    "win lottery claim prize",
    "urgent offer win money",
    "free cash bonus now",
    "buy meds online cheap",
    "meeting schedule tomorrow",
    "project discussion meeting",
    "please review the report",
    "team meeting agenda today",
    "project deadline discussion",
    "review the project document",
    "schedule a meeting tomorrow",
    "please send the report",
    "discussion on project update",
    "team sync meeting notes"
]

labels = np.array([
    1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0
])
```

Where:

* `1 = spam`
* `0 = non-spam`

### Task

Implement:

```python
naive_bayes_mle_spam()
```

### Steps

1. Tokenize the emails
2. Build the vocabulary
3. Compute class priors
4. Compute word probabilities using **MLE**
5. Predict the class of:

```python
test_email = "win cash prize now"
```

### Important

* **Do not use Laplace smoothing**
* If a word never appears in a class, its probability is 0

### Return

```python
priors, word_probs, prediction
```

Where:

* `priors` is a dictionary
* `word_probs` is a nested dictionary
* `prediction` is `0` or `1`

---

## Q2 — K-Nearest Neighbors on Iris

Implement **KNN from scratch** on the Iris dataset.

### Task

Implement:

```python
knn_iris(k=3, test_size=0.2, seed=0)
```

### Steps

1. Load the Iris dataset
2. Split into train/test sets
3. For each test example:

   * compute Euclidean distance to all training examples
   * find the `k` nearest neighbors
   * predict by majority vote
4. Compute train and test accuracy

### Return

```python
train_accuracy, test_accuracy, predictions
```

Where:

* `train_accuracy` is a float
* `test_accuracy` is a float
* `predictions` is a numpy array of predicted labels on the test set

---

## Rules

* Do not rename functions
* Do not change return formats
* Do not use:

  * `sklearn.naive_bayes`
  * `sklearn.neighbors.KNeighborsClassifier`

---

## Installation

```bash
pip install numpy scikit-learn pytest
```

---

## Run Tests

```bash
pytest -q
```
