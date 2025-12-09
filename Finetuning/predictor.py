import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = "dbert_finetuning"
DEV_JSON = "dev.json"
OUTPUT_JSONL = "predictions_fin.jsonl"

def load_data(json_file):
    """load a json file"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = list(data.values()) if isinstance(data, dict) else data
    df = pd.DataFrame(items)
    return df

def prepare_text(row):
    """Prepare text for prediciton"""
    meaning_text = row["judged_meaning"].strip() if row["judged_meaning"] else row["example_sentence"].strip()
    return f"Word: {row['homonym']}. Context: {row['precontext']} {row['sentence']} {row['ending']} Meaning: {meaning_text}".strip()

def format_output(predictions, jsonl_file):
    """Save predictions in the desired form"""
    output = []
    for idx, pred in enumerate(predictions):
        output.append({"id": str(idx), "prediction": float(pred)})
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

def main():
    print("Loading regression SBERT model...")
    model = SentenceTransformer(MODEL_PATH)

    print("Loading dev data...")
    df_dev = load_data(DEV_JSON)
    texts = df_dev.apply(prepare_text, axis=1).tolist()

    print(f"Encoding {len(texts)} examples...")
    preds = model.encode(texts, convert_to_numpy=True, batch_size=32).flatten()

    # Rescale predictions from 0–1 → 1–5
    preds_rescaled = preds * 4.0 + 1.0

    # rounding
    preds_rescaled = np.round(preds_rescaled)

    print(f"Saving predictions to {OUTPUT_JSONL}...")
    format_output(preds_rescaled, OUTPUT_JSONL)
    print("✓ Done.")

if __name__ == "__main__":
    main()
