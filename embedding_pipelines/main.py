import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from embedding_pipeline import run_classification_pipeline, run_regression_pipeline


def load_embeddings(file_path, normalize=False):
    df = pd.read_csv(file_path, header=0)
    if normalize:
        normalizer = Normalizer()
        return normalizer.fit_transform(df.values)
    return df.values  # normalize?


def load_labels(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [data[item]["average"] for item in data]


def format_output(predictions, jsonl_file):
    # save predictions in format {"id": "1", "prediction": 3} as jsonl
    output = []
    for idx, pred in enumerate(predictions):
        output.append({"id": str(idx), "prediction": round(pred)})
    with open(jsonl_file, 'w') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')


def main():
    X_train = load_embeddings('../e5smallv2-alltexts/embeddings_out_train/all_texts.csv', normalize=False)
    X_dev = load_embeddings('../e5smallv2-alltexts/embeddings_out_dev/all_texts.csv', normalize=False)
    y_train = load_labels('../Data/train.json')
    y_dev = load_labels('../Data/dev.json')

    print(len(X_train), len(y_train))
    

    # use different embeddings and cosine sims? Bei
    text_embeddings = load_embeddings('../e5smallv2-separate/embeddings_out_dev/text_only.csv')
    meaning_embeddings = load_embeddings('../e5smallv2-separate/embeddings_out_dev/meaning_only.csv')

    format_output(run_regression_pipeline(X_train, y_train, X_dev), 'predictions_pipeline.jsonl')
    format_output(run_classification_pipeline(X_train, y_train, X_dev), 'predictions_pipeline.jsonl')
    

if __name__ == "__main__":
    main()