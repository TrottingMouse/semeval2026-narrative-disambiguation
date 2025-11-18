from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer


import json
import pandas as pd


def load_embeddings(file_path, normalize=False):
    df = pd.read_csv(file_path, header=0)
    if normalize:
        normalizer = Normalizer()
        return normalizer.fit_transform(df.values)
    return df.values  # normalize?


def load_labels(file_path, rounded=False):
    with open(file_path, 'r') as f:
        data = json.load(f)
    if rounded:
        return [round(data[item]["average"]) for item in data]
    return [data[item]["average"] for item in data]


def format_output(predictions, jsonl_file):
    # save predictions in format {"id": 1, "prediction": 3} as jsonl
    output = []
    for idx, pred in enumerate(predictions):
        output.append({"id": str(idx), "prediction": round(pred)})
    with open(jsonl_file, 'w') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')


X_train = load_embeddings('embeddings_out_train/text_plus_expl.csv', normalize=True)
X_dev = load_embeddings('embeddings_out_dev/text_plus_expl.csv', normalize=True)
y_train = load_labels('Data/train.json', rounded=False)
y_dev = load_labels('Data/dev.json', rounded=False)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_dev)
score = pipeline.score(X_dev, y_dev)
format_output(predictions, 'predictions.jsonl')