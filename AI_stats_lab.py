import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
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

    test_email = "win cash prize now"

    # 1. Tokenize
    tokenized_texts = [text.split() for text in texts]

    # 2. Vocabulary
    vocab = set(word for text in tokenized_texts for word in text)

    # 3. Priors
    priors = {
        0: np.mean(labels == 0),
        1: np.mean(labels == 1)
    }

    # 4. Word counts per class
    word_counts = {
        0: Counter(),
        1: Counter()
    }

    total_words = {
        0: 0,
        1: 0
    }

    for text, label in zip(tokenized_texts, labels):
        word_counts[label].update(text)
        total_words[label] += len(text)

    # 5. Word probabilities (MLE)
    word_probs = {
        0: {},
        1: {}
    }

    for c in [0, 1]:
        for word in vocab:
            word_probs[c][word] = word_counts[c][word] / total_words[c] if total_words[c] > 0 else 0

    # 6. Prediction (log probabilities to avoid underflow)
    test_words = test_email.split()

    log_probs = {}

    for c in [0, 1]:
        log_prob = np.log(priors[c])

        for word in test_words:
            prob = word_probs[c].get(word, 0)

            if prob > 0:
                log_prob += np.log(prob)
            else:
                # since no smoothing, unseen word → probability 0 → log = -inf
                log_prob += -np.inf

        log_probs[c] = log_prob

    prediction = max(log_probs, key=log_probs.get)

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):

    # 1. Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 3. Euclidean distance
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # 4. KNN prediction
    def predict(X_train, y_train, x_test, k):
        distances = []

        for i in range(len(X_train)):
            dist = euclidean_distance(x_test, X_train[i])
            distances.append((dist, y_train[i]))

        # sort by distance
        distances.sort(key=lambda x: x[0])

        # get k nearest labels
        k_neighbors = [label for _, label in distances[:k]]

        # majority vote
        return Counter(k_neighbors).most_common(1)[0][0]

    # 5. Predictions
    train_preds = np.array([predict(X_train, y_train, x, k) for x in X_train])
    test_preds = np.array([predict(X_train, y_train, x, k) for x in X_test])

    # 6. Accuracy
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    return train_accuracy, test_accuracy, test_preds
