# Embedding: E5-small-v2 – nur Text, nur Bedeutung und Text + Erklärung

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

    # Drei Textvarianten aufbauen
    df["text_only"] = (
        (df["precontext"] + " " + df["sentence"] + " " + df["ending"])
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["meaning_only"] = (
        df["judged_meaning"]
        .where(df["judged_meaning"].str.len() > 0, df["example_sentence"])
        .str.strip()
    )
    df["text_plus_expl"] = (
        df["text_only"] + " Explanation: " + df["meaning_only"]
    ).str.strip()

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
        "text_only", "meaning_only", "text_plus_expl"
    ]
    df[meta_cols].to_csv(out_dir / "meta.csv", index=False, encoding="utf-8")

    # Embeddings speichern (eine Datei pro Variante)
    for name, arr in embs.items():
        pd.DataFrame(arr).to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")

    print(f"[Save] CSV-Dateien gespeichert unter {out_dir.resolve()}")


# 4. E5-Helfer

def apply_e5_prefixes(df: pd.DataFrame, mode: str):
    # E5-Modelle erwarten Prefixes.

    mode = mode.lower()
    if mode == "none":
        print("[E5] Prefixing ist deaktiviert (mode=none).")
        return df

    if mode not in {"passage", "query"}:
        print(f"[E5] Unbekannter e5_mode '{mode}', benutze 'passage' als Fallback.")
        mode = "passage"

    prefix = f"{mode}: "
    cols = ["text_only", "meaning_only", "text_plus_expl"]

    for col in cols:
        if col in df.columns:
            df[col] = prefix + df[col].astype(str)

    return df


# 5. Argumente (CLI)

def parse_args():
    p = argparse.ArgumentParser(description="Embeddings (3 Varianten) mit E5-small-v2-Support")
    p.add_argument("--input_json", default="train.json")
    # Standard jetzt: E5-small-v2
    p.add_argument("--model_name", default="intfloat/e5-small-v2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--out_dir", default="embeddings_out")
    p.add_argument("--no_save", action="store_true")

    # E5-spezifisch:
    # passage = für Dokument-/Passage-Embeddings (Standard)
    # query   = falls du Queries einbetten willst
    # none    = falls du manuell Prefixes gebaut hast
    p.add_argument(
        "--e5_mode",
        default="passage",
        help="E5-Prefix-Modus: 'passage', 'query' oder 'none' (Standard: passage)",
    )

    return p.parse_args()


# 6. Hauptlogik

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




# Einstiegspunkt

if __name__ == "__main__":
    main()
