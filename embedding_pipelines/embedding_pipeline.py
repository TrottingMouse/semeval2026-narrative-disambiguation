from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import json
import pandas as pd





def train_binary_classifier(sklearn_classifier, X_train, y_train, class_label: int):
    # Convert to binary labels
    y_train_binary = [1 if round(label) == class_label else 0 for label in y_train]
    sklearn_classifier.fit(X_train, y_train_binary)
    return sklearn_classifier
    



def run_regression_pipeline(X_train, y_train, X_dev, model=LinearRegression()):
    """Train a simple pipeline and return predictions."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_dev)
    return preds

def run_classification_pipeline(X_train, y_train, X_dev, model=SVC(kernel='linear', probability=True)):
    """Train a simple classification pipeline and return predictions."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('model', model)
    ])
    rounded_y_train = [round(label) for label in y_train]
    pipeline.fit(X_train, rounded_y_train)
    preds = pipeline.predict(X_dev)
    return preds

#format_output(predictions, 'predictions.jsonl')

# elaborate pipeline with binary classifiers for each class
def run_binary_classifiers(X_train, X_dev, y_train):
    classifiers = []
    for class_label in range(1, 6):
        svm_classifier = SVC(kernel='linear', probability=True)
        trained_classifier = train_binary_classifier(svm_classifier, X_train, y_train, class_label)
        classifiers.append(trained_classifier)
# Get probabilities for each class
    probabilities = []
    for clf in classifiers:
        probs = clf.predict_proba(X_dev)[:, 1]  # Probability of the positive class
        probabilities.append(probs)
# Transpose to get a list of probabilities for each instance
    probabilities = list(zip(*probabilities))
# Predict the sum of probabilities rounded to nearest integer
    final_predictions = []
    for probs in probabilities:
        pred = 0
        for class_label, prob in enumerate(probs, start=1):
            pred += prob * class_label
        final_predictions.append(round(pred))
    return final_predictions

#final_predictions = run_binary_classifiers(X_train, X_dev, y_train)
#format_output(final_predictions, 'predictions_binary_classifiers.jsonl')





# compute cosine similarity
def compute_correlation(true_labels, text_embeddings, meaning_embeddings):
    cosine_sims = np.diag(cosine_similarity(text_embeddings, meaning_embeddings))

# compute correlation with true labels
    true_labels = load_labels('Data/train.json', rounded=False)
    correlation = np.corrcoef(cosine_sims, true_labels)[0, 1]
    return correlation

#correlation = compute_correlation(load_labels, text_embeddings, meaning_embeddings)


#print(f"Correlation: {correlation}")