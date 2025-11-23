# main.py
from train_experiment import train_experiment
from train_experiment import save_confusion_matrix

EXPERIMENTS = [
    # 1) BERT + PCA 200 (baseline fuerte)
    {
        "name": "bert_pca200",
        "embedding_type": "bert",
        "pca_dim": 200,
        "epochs": 100,
        "batch_size": 32,
        "lr": 1e-4,
    },

    # 2) TF-IDF unigrams + bigrams
    {
        "name": "tfidf_1_2_pca200",
        "embedding_type": "tfidf",
        "pca_dim": 200,
        "epochs": 100,
        "batch_size": 64,
        "lr": 1e-3,
        "vectorizer_params": {
            "max_features": 10000,
            "ngram_range": [1, 2],
            "min_df": 5,
            "max_df": 0.95,
        },
    },

    # 3) Bag-of-Words unigrams
    {
        "name": "bow_1_pca200",
        "embedding_type": "bow",
        "pca_dim": 200,
        "epochs": 100,
        "batch_size": 64,
        "lr": 1e-3,
        "vectorizer_params": {
            "max_features": 10000,
            "ngram_range": [1, 1],
            "min_df": 5,
            "max_df": 0.95,
        },
    },
]


if __name__ == "__main__":
    resultados = []
    for cfg in EXPERIMENTS:
        result = train_experiment(cfg)
        resultados.append(result)

    print("\n=== RESUMEN FINAL ===")
    for r in resultados:
        name = r["name"]
        f1 = r["result"]["final_f1"]["f1_global"]
        print(f"- {name}: F1_final={f1:.4f}")

    # Elegir mejor experimento por F1_final
    best = max(resultados, key=lambda x: x["result"]["final_f1"]["f1_global"])

    print("\n=== MEJOR MODELO: {} ===".format(best["name"]))
    print("F1 =", best["result"]["final_f1"]["f1_global"])

    save_confusion_matrix(
        y_true=best["y_true"],
        y_pred=best["y_pred"],
        labels=best["labels"],
        filename=f"./output/confusion_matrix_{best['name']}.png"
    )

    print(f"Matriz de confusión guardada en ./output/confusion_matrix_{best['name']}.png")

