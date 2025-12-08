import pandas as pd

df = pd.read_json("results/predictions.jsonl", orient='records', lines=True)
df['predictions'] = df['predictions'].astype(int)
df = df.rename(columns={
    "index": "id",
    "predictions": "prediction"
})

df.to_json("results/predictions.jsonl", orient='records', lines=True)


