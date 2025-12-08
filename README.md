# Semeval26 Task 5
## Evaluation
Refer to format and evaluation script here:

https://github.com/Janosch-Gehring/semeval26-05-scripts


## Methods

* Jan, Sarah: (Finetune) Embeddings -> Lineare Regression -> runden
* Jan, Sarah: (Finetune) Embeddings -> Klassifikation
* Daniel: Embeddings -> 5 binäre Klassifikatoren -> durchschnittliches gewichtetes Gesamtergebnis
* Julio: meaning an Wort -> Surprisal & Similarity Score -> 
* Matthias: ähnliche Begriffe dazu -> andere Methoden


## Current results
Baseline (random):

{"accuracy": 0.44727891156462585, "spearman": -0.08018581545573154}

Baseline (majority):

{"accuracy": 0.5697278911564626, "spearman": NaN}


Embeddings (meaning included in text) + Linear Regression:

{"accuracy": 0.5459183673469388, "spearman": 0.01799331520799271}

Embeddings (meaning included in text) + SVC:

{"accuracy": 0.5357142857142857, "spearman": 0.05264041844071247}

Correlation for label average vs cosine similarity:

0.26805305386871225

Surprisal:

{"accuracy": 0.5697278911564626, "spearman": -0.11527990914334803}

{"accuracy": 0.477891156462585, "spearman": -0.0819439441233343}

Finetuning:

{"accuracy": 0.5918367346938775, "spearman": 0.2101613315469357}
