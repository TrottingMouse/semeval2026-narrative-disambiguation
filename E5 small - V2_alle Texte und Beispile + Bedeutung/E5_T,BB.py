import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Hilfsfunktion: JSON-Datei laden und vorbereiten

def load_json(path: str) -> pd.DataFrame:
    p = Path(path)
    with open(p, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    items = list(raw.values()) if isinstance(raw, dict) else raw
    df = pd.DataFrame(items)

    for col in ["precontext", "sentence", "ending", "judged_meaning", "example_sentence"]:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["text_only"] = (
        (df["precontext"] + " " + df["sentence"] + " " + df["ending"])
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # E5 benötigt Prefixes
    df["example_plus_meaning"] = (
        "passage: Example: " + df["example_sentence"].str.strip() +
        " Meaning: " + df["judged_meaning"].str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    df["all_texts"] = (
        "passage: " + df["precontext"] + " " + df["sentence"] + " " +
        df["ending"] + " " + df["judged_meaning"] + " " + df["example_sentence"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return df

# 2. Embeddings berechnen

def embed(series: pd.Series, model: SentenceTransformer, bs: int, norm: bool) -> np.ndarray:
    return model.encode(
        series.tolist(),
        batch_size=bs,
        convert_to_numpy=True,
        normalize_embeddings=norm,
        show_progress_bar=True,
    )

# 3. Ergebnisse speichern

def save_csv(df: pd.DataFrame, embs: dict[str, np.ndarray], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = [
        "precontext", "sentence", "ending",
        "judged_meaning", "example_sentence",
        "text_only", "example_plus_meaning", "all_texts"
    ]
    df[meta_cols].to_csv(out_dir / "meta.csv", index=False, encoding="utf-8")

    for name, arr in embs.items():
        pd.DataFrame(arr).to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")

    print(f"[Save] CSV-Dateien gespeichert unter {out_dir.resolve()}")

# 4. Hauptlogik

def main():
    parser = argparse.ArgumentParser(description="Embeddings mit E5-small-v2")
    parser.add_argument("--input_json", default="train.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--out_dir", default="embeddings_out")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    df = load_json(args.input_json)

    # E5-small-v2
    model = SentenceTransformer("intfloat/e5-small-v2")
    norm = not args.no_normalize

    embs = {
        "example_plus_meaning": embed(df["example_plus_meaning"], model, args.batch_size, norm),
        "all_texts": embed(df["all_texts"], model, args.batch_size, norm),
    }

    if not args.no_save:
        save_csv(df, embs, Path(args.out_dir))

    print("✓ Fertig.")

if __name__ == "__main__":
    main()
