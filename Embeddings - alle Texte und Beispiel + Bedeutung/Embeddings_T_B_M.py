# Embedding: Beispiel + Bedeutung UND alle Texte zusammen

import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer


# 1. Hilfsfunktion: JSON-Datei laden und vorbereiten

def load_json(path: str) -> pd.DataFrame:
    # Lädt eine JSON-Datei und erstellt alle notwendigen Textvarianten.
    p = Path(path)
    with open(p, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    # Unterstützt sowohl dict als auch list
    items = list(raw.values()) if isinstance(raw, dict) else raw
    df = pd.DataFrame(items)

    # Fehlende Spalten absichern
    for col in ["precontext", "sentence", "ending", "judged_meaning", "example_sentence"]:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Basis-Text (für "alle Texte zusammen")
    df["text_only"] = (
        (df["precontext"] + " " + df["sentence"] + " " + df["ending"])
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Variante 1: Beispiel + Bedeutung
    df["example_plus_meaning"] = (
        "Example: " + df["example_sentence"].str.strip() +
        " Meaning: " + df["judged_meaning"].str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Variante 2: alle Texte zusammen
    df["all_texts"] = (
        df["precontext"] + " " +
        df["sentence"] + " " +
        df["ending"] + " " +
        df["judged_meaning"] + " " +
        df["example_sentence"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return df


# 2. Embeddings berechnen

def embed(series: pd.Series, model: SentenceTransformer, bs: int, norm: bool) -> np.ndarray:
    # Berechnet Embeddings für eine Textspalte mit Fortschrittsanzeige.
    return model.encode(
        series.tolist(),
        batch_size=bs,
        convert_to_numpy=True,
        normalize_embeddings=norm,
        show_progress_bar=True,
    )


# 3. Ergebnisse speichern

def save_csv(df: pd.DataFrame, embs: dict[str, np.ndarray], out_dir: Path):
    # Speichert Metadaten + Embeddings als CSV-Dateien im angegebenen Ordner.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metadaten speichern
    meta_cols = [
        "precontext", "sentence", "ending",
        "judged_meaning", "example_sentence",
        "text_only", "example_plus_meaning", "all_texts"
    ]
    df[meta_cols].to_csv(out_dir / "meta.csv", index=False, encoding="utf-8")

    # Embeddings speichern (eine Datei pro Variante)
    for name, arr in embs.items():
        pd.DataFrame(arr).to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")

    print(f"[Save] CSV-Dateien gespeichert unter {out_dir.resolve()}")


# 5. Argumente (CLI)

def parse_args():
    p = argparse.ArgumentParser(description="Embeddings (Beispiel+Bedeutung & alle Texte)")
    p.add_argument("--input_json", default="train.json")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--out_dir", default="embeddings_out")
    p.add_argument("--no_save", action="store_true")
    return p.parse_args()


# 6. Hauptlogik
def main():
    args = parse_args()
    df = load_json(args.input_json)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    norm = not args.no_normalize

    # Embeddings berechnen (2 Varianten)
    embs = {
        "example_plus_meaning": embed(df["example_plus_meaning"], model, args.batch_size, norm),
        "all_texts": embed(df["all_texts"], model, args.batch_size, norm),
    }

    # Ergebnisse speichern 
    if not args.no_save:
        save_csv(df, embs, Path(args.out_dir))

    print("✓ Fertig.")


# Einstiegspunkt

if __name__ == "__main__":
    main()

