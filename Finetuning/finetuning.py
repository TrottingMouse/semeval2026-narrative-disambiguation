import json
import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models

def load_json(path: str) -> pd.DataFrame:
    """load a json file"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = list(raw.values()) if isinstance(raw, dict) else raw
    df = pd.DataFrame(items)
    required = ["homonym", "judged_meaning", "example_sentence", "precontext", "sentence", "ending", "average", "sample_id"]
    for c in required:
        if c not in df:
            df[c] = ""
    df["average"] = pd.to_numeric(df["average"], errors="coerce")
    return df

def build_examples(df: pd.DataFrame) -> list:
    """Build examples and prepare for regression head"""
    examples = []
    for i, row in df.iterrows():
        meaning_text = row["judged_meaning"].strip() if row["judged_meaning"] else row["example_sentence"].strip()
        if not meaning_text:
            continue
        text = f"Word: {row['homonym']}. Context: {row['precontext']} {row['sentence']} {row['ending']} Meaning: {meaning_text}".strip()
        # Scale label 1–5 → 0–1 and wrap in list to match output [batch,1]
        label_scaled = [(float(row["average"]) - 1.0) / 4.0]
        examples.append(InputExample(texts=[text], label=label_scaled))
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="train.json")
    parser.add_argument("--dev_json", default="dev.json")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--save_dir", default="dbert_finetuning")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)  # can adjust
    args = parser.parse_args()

    print("Loading train data...")
    df_train = load_json(args.train_json)
    train_examples = build_examples(df_train)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    print("Loading dev data...")
    df_dev = load_json(args.dev_json)
    dev_examples = build_examples(df_dev)
    dev_loader = DataLoader(dev_examples, shuffle=False, batch_size=args.batch_size)

    print("Loading base model:", args.model_name)
    model = SentenceTransformer(args.model_name)

    # unfreeze distilbert
    for param in model.parameters():
        param.requires_grad = True

    # Add regression head
    regression_head = models.Dense(
        in_features=model.get_sentence_embedding_dimension(),
        out_features=1,
        activation_function=None
    )
    model.add_module('regression_head', regression_head)

    # MSE loss
    train_loss = losses.MSELoss(model=model)

    warmup_steps = int(0.05 * len(train_loader) * args.epochs)
    print(f"Warmup steps: {warmup_steps}")

    print("Training regression head with unfrozen base...")
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.lr},
        show_progress_bar=True
    )

    # Save model
    print("Saving to:", args.save_dir)
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    model.save(args.save_dir)
    print("✓ Done.")

if __name__ == "__main__":
    main()


