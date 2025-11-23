# train_experiment.py
import os
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


from neural_networks.base_model import BaseModel
from evaluacion import fscore
import embeddings
import utils


def save_confusion_matrix(y_true, y_pred, labels, filename="./output/confusion_matrix.png"):
    """
    Genera y guarda una matriz de confusión como imagen PNG.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de Confusión (Mejor modelo: BOW + PCA 200)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()

def train_experiment(config):
    """
    Lanza un experimento completo:
      - Genera / carga embeddings (bert / tfidf / bow)
      - Aplica PCA
      - Entrena la red neuronal
      - Evalúa F1 en train y test
      - Guarda resultados en ./experimentos/<name>.json
    """
    cfg = utils.get_complete_config(config)

    SEED = cfg["seed"]
    PCA_DIM = cfg["pca_dim"]
    LR = cfg["lr"]
    WEIGHT_DECAY = cfg["weight_decay"]
    EPOCHS = cfg["epochs"]
    BATCH_SIZE = cfg["batch_size"]
    F1_MODE = cfg["f1_mode"]
    EMBEDDING_TYPE = cfg["embedding_type"]
    VECTORIZER_PARAMS = cfg["vectorizer_params"]

    DATA_PATH = cfg["data_path"]
    TEXT_COL = cfg["text_col"]
    ID_COL = cfg["id_col"]
    LABEL_COL = cfg["label_col"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print(f"\n=== Experimento: {cfg['name']} ===")
    print(f" - Embedding: {EMBEDDING_TYPE}")
    print(f" - PCA dim:  {PCA_DIM}")
    print(f" - LR:       {LR}")
    print(f" - Batch:    {BATCH_SIZE}")
    print(f" - Epochs:   {EPOCHS}")

    # 1. Cargar / generar embeddings
    base_name = os.path.splitext(os.path.basename(DATA_PATH))[0]
    embeddings_file = os.path.join("./output", f"{base_name}_{EMBEDDING_TYPE}_embeddings.csv")

    if os.path.isfile(embeddings_file):
        print(f"[train_experiment] Cargando embeddings desde {embeddings_file}")
        df = pd.read_csv(embeddings_file)
    else:
        print("[train_experiment] Generando embeddings...")
        df = embeddings.embeddings(
            DATA_PATH,
            method=EMBEDDING_TYPE,
            output_dir="./output",
            vectorizer_params=VECTORIZER_PARAMS,
            text_col=TEXT_COL,
            id_col=ID_COL,
            label_col=LABEL_COL,
        )

    # 2. Separar X e y
    if LABEL_COL not in df.columns:
        raise ValueError(f"No se encontró la columna de etiqueta '{LABEL_COL}' en el DataFrame de embeddings.")

    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    # 3. Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"[train_experiment] Número de clases: {num_classes}")

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=SEED,
        stratify=y_encoded,
    )

    # Guardamos ids por si luego quieres analizarlos
    if ID_COL in X_train.columns:
        ids_train = X_train[ID_COL].reset_index(drop=True)
        X_train_features = X_train.drop(columns=[ID_COL])
    else:
        ids_train = None
        X_train_features = X_train

    if ID_COL in X_test.columns:
        ids_test = X_test[ID_COL].reset_index(drop=True)
        X_test_features = X_test.drop(columns=[ID_COL])
    else:
        ids_test = None
        X_test_features = X_test

    # 5. PCA
    n_features = X_train_features.shape[1]
    n_components = min(PCA_DIM, n_features)
    print(f"[train_experiment] Aplicando PCA a {n_features} features -> {n_components} componentes")
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train_features.values)
    X_test_reduced = pca.transform(X_test_features.values)

    # 6. Tensors
    X_train_t = torch.tensor(X_train_reduced, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_test_t = torch.tensor(X_test_reduced, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE)

    # 7. Modelo
    model = BaseModel(input_dim=n_components, output_dim=num_classes).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 8. Entrenamiento
    history = []
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        # --- Fase de entrenamiento ---
        model.train()
        train_loss = 0.0

        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Fase de evaluación ---
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.inference_mode():
            for Xb, yb in test_loader:
                logits = model(Xb)
                loss = loss_fn(logits, yb)
                test_loss += loss.item() * Xb.size(0)

                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())

        test_loss /= len(test_loader.dataset)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # F1 en train y test
        train_preds = torch.argmax(model(X_train_t), dim=1).cpu()
        f1_train = fscore(num_classes, y_train_t.cpu(), train_preds, le, F1_MODE)
        f1_test = fscore(num_classes, all_labels, all_preds, le, F1_MODE)

        if f1_test["f1_global"] > best_f1:
            best_f1 = f1_test["f1_global"]
            best_epoch = epoch

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"[{cfg['name']}] Epoch {epoch:03d} | "
                f"loss_train={train_loss:.4f} | loss_test={test_loss:.4f} | "
                f"F1_test={f1_test['f1_global']:.4f}"
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "f1_train": f1_train,
                "f1_test": f1_test,
            }
        )

    final_result = {
        "config": cfg,
        "final_f1": f1_test,        # último epoch
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "history": history,
    }

    os.makedirs("./experimentos", exist_ok=True)
    out_file = f"./experimentos/{cfg['name']}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)

    print(
        f"\n✅ Experimento '{cfg['name']}' finalizado | "
        f"F1_final={f1_test['f1_global']:.4f} | "
        f"F1_mejor={best_f1:.4f} (epoch {best_epoch})"
    )

    return {
        "result": final_result,
        "name": cfg["name"],
        "y_true": y_test_t.cpu().numpy(),
        "y_pred": all_preds.cpu().numpy(),
        "labels": le.classes_
    }

