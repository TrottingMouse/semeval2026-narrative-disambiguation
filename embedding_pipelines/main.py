import json
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from embedding_pipeline import compute_correlation, run_binary_classifiers, run_classification_pipeline, run_regression_pipeline


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
    dir_path = ""
    # X_train = load_embeddings('../e5smallv2-alltexts/embeddings_out_train/all_texts.csv', normalize=False)
    # X_dev = load_embeddings('../e5smallv2-alltexts/embeddings_out_dev/all_texts.csv', normalize=False)
    # y_train = load_labels('../Data/train.json')
    # y_dev = load_labels('../Data/dev.json')

    # print(len(X_train), len(y_train))

    for name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, name)
        X_train = load_embeddings(os.path.join(file_path, "Embeddings_BB_T/embeddings_out_train/all_texts.csv"))
        X_dev = load_embeddings(os.path.join(file_path, "Embeddings_BB_T/embeddings_out_dev/all_texts.csv"))
        y_train = load_labels('../Data/train.json')
        format_output(run_regression_pipeline(X_train, y_train, X_dev), f'predictions_{name}.jsonl')

    

    # use different embeddings and cosine sims?
    # text_embeddings = load_embeddings('../e5smallv2-separate/embeddings_out_train/text_only.csv')
    # meaning_embeddings = load_embeddings('../e5smallv2-separate/embeddings_out_train/meaning_only.csv')

    #format_output(run_regression_pipeline(X_train, y_train, X_dev), 'predictions_pipeline.jsonl')
    #format_output(run_classification_pipeline(X_train, y_train, X_dev), 'predictions_pipeline.jsonl')
    # compute correlation
    # correlation = compute_correlation(y_train, text_embeddings, meaning_embeddings)
    # print(f"Correlation: {correlation}")
    ä#format_output(run_binary_classifiers(X_train, X_dev, y_train), 'predictions_binary_classifiers_with_meta.jsonl')
    

if __name__ == "__main__":
    main()