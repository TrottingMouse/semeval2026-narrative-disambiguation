from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import json


def load_embeddings(file_path):
    pass


def load_labels(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [data[item]["average"] for item in data]


X_train = load_embeddings('Embeddings/train_embeddings.csv')
X_dev = load_embeddings('Embeddings/dev_embeddings.csv')
y_train = load_labels('Data/train.json')
y_dev = load_labels('Data/dev.json')


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_dev)
print(predictions)
