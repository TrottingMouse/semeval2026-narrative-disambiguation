# Semeval26 Task 5
## Evaluation
Refer to format and evaluation script here:

https://github.com/Janosch-Gehring/semeval26-05-scripts


## Methods

* Jan, Sarah: (Finetune) Embeddings -> Lineare Regression -> runden
* Jan, Sarah: (Finetune) Embeddings -> Klassifikation
* Daniel: Embeddings -> 5 binäre Klassifikatoren -> durchschnittliches gewichtetes Gesamtergebnis | neu: 4 binäre Klassifikatoren mit größer gleich Klasse (Ausnutzen der Ordnung)
* Julio: meaning an Wort -> Surprisal & Similarity Score -> 
* Matthias: ähnliche Begriffe dazu -> andere Methoden


## Current results
Baseline (random):

{"accuracy": 0.44727891156462585, "spearman": -0.08018581545573154}

Baseline (majority):

{"accuracy": 0.5697278911564626, "spearman": NaN}


Embeddings (meaning included in text) + Linear Regression:

* all-Mini_L6-v2: {"accuracy": 0.5255102040816326, "spearman": 0.00656551692026338}
* all-mpnet-base-v2: {"accuracy": 0.564625850340136, "spearman": 0.11322736541410187}
* BAAlbge-base-en-v1.5: {"accuracy": 0.5272108843537415, "spearman": 0.014350656246544052}
* e5-small-v2: {"accuracy": 0.5357142857142857, "spearman": 0.039004202747490446}
* e5-base-v2: {"accuracy": 0.5068027210884354, "spearman": 0.013937386572863268}
* nomic-embed-text: {"accuracy": 0.5714285714285714, "spearman": 0.01871672965755985}

Keine Ahnung woher: {"accuracy": 0.5459183673469388, "spearman": 0.01799331520799271}

Embeddings (meaning included in text) + SVC:

{"accuracy": 0.5357142857142857, "spearman": 0.05264041844071247}

Correlation for label average vs cosine similarity:

0.26805305386871225

Binary classifiers without meta classifier:

{"accuracy": 0.5255102040816326, "spearman": -0.032193215004679726}

Binary classifiers with meta classifier:

{"accuracy": 0.54421768707483, "spearman": 0.009013676075603749}

Surprisal:

{"accuracy": 0.5697278911564626, "spearman": -0.11527990914334803}

{"accuracy": 0.477891156462585, "spearman": -0.0819439441233343}

Finetuning:

{"accuracy": 0.5918367346938775, "spearman": 0.2101613315469357}
