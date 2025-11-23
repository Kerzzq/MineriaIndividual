# embeddings.py
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import os


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # Modelo médico
EMBEDDING_DIM = 768  # Dimensión típica del modelo BERT-base


def cargar_dataset(csv_path, text_col="open_response", id_col="newid", label_col="gs_text34"):
    """
    Carga el dataset y devuelve listas con los textos, sus IDs y etiquetas.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    for col in [text_col, id_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"El CSV debe contener la columna '{col}'.")

    textos = df[text_col].astype(str).fillna("").tolist()
    ids = df[id_col].tolist()
    labels = df[label_col].tolist()
    return ids, textos, labels


def generar_embeddings_bert(textos, model_name=MODEL_NAME):
    """
    Genera embeddings BERT (Bio_ClinicalBERT) a partir de una lista de textos.
    Devuelve una matriz numpy de tamaño (N, D).
    """
    print(f"\n[Capa de embeddings] Cargando modelo BERT: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[Capa de embeddings] Generando embeddings BERT para {len(textos)} instancias...")
    embeddings = model.encode(textos, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def generar_embeddings_tfidf(textos, vectorizer_params=None):
    """
    Genera embeddings TF-IDF a partir de una lista de textos.
    vectorizer_params puede incluir:
      - max_features (int)
      - ngram_range (lista [1,2] que convertimos a tupla)
      - min_df, max_df, etc.
    """
    if vectorizer_params is None:
        vectorizer_params = {}

    # Normalizamos ngram_range si viene como lista desde JSON
    ngram_range = vectorizer_params.get("ngram_range", [1, 1])
    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)

    params = {
        "max_features": vectorizer_params.get("max_features", 10000),
        "ngram_range": ngram_range,
        "min_df": vectorizer_params.get("min_df", 2),
        "max_df": vectorizer_params.get("max_df", 0.95),
    }

    print(f"\n[Capa de embeddings] Generando TF-IDF con parámetros: {params}")
    vectorizer = TfidfVectorizer(**params)
    X = vectorizer.fit_transform(textos)
    return X.toarray()


def generar_embeddings_bow(textos, vectorizer_params=None):
    """
    Genera embeddings Bag-of-Words (frecuencias) a partir de una lista de textos.
    """
    if vectorizer_params is None:
        vectorizer_params = {}

    ngram_range = vectorizer_params.get("ngram_range", [1, 1])
    if isinstance(ngram_range, list):
        ngram_range = tuple(ngram_range)

    params = {
        "max_features": vectorizer_params.get("max_features", 10000),
        "ngram_range": ngram_range,
        "min_df": vectorizer_params.get("min_df", 2),
        "max_df": vectorizer_params.get("max_df", 0.95),
    }

    print(f"\n[Capa de embeddings] Generando Bag-of-Words con parámetros: {params}")
    vectorizer = CountVectorizer(**params)
    X = vectorizer.fit_transform(textos)
    return X.toarray()


def embeddings(
    data_path,
    method="bert",
    output_dir="./output",
    vectorizer_params=None,
    text_col="open_response",
    id_col="newid",
    label_col="gs_text34",
):
    """
    Punto de entrada unificado.
    Genera embeddings usando el método indicado:
      - 'bert'
      - 'tfidf'
      - 'bow'

    Devuelve un DataFrame con:
      - columna 'id'
      - columnas de features numéricos
      - columna de etiqueta (label_col, por defecto 'gs_text34')
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(data_path))[0]
    method = method.lower()

    output_csv = os.path.join(output_dir, f"{base_name}_{method}_embeddings.csv")

    # Si ya existe, lo cargamos directamente (evita recalcular)
    if os.path.exists(output_csv):
        print(f"[Capa de embeddings] Encontrado fichero existente: {output_csv}")
        return pd.read_csv(output_csv)

    # Cargar datos
    ids, textos, labels = cargar_dataset(data_path, text_col=text_col, id_col=id_col, label_col=label_col)

    # Seleccionar método de vectorización
    if method == "bert":
        X = generar_embeddings_bert(textos)
    elif method == "tfidf":
        X = generar_embeddings_tfidf(textos, vectorizer_params)
    elif method == "bow":
        X = generar_embeddings_bow(textos, vectorizer_params)
    else:
        raise ValueError(f"Método de embeddings desconocido: {method}")

    # Crear DataFrame con resultados
    df_out = pd.DataFrame(X)
    df_out.insert(0, "id", ids)
    df_out[label_col] = labels

    # Guardar CSV
    df_out.to_csv(output_csv, index=False)
    print(f"\n[Capa de embeddings] Embeddings guardados correctamente en '{output_csv}'")

    return df_out


if __name__ == "__main__":
    # Ejemplo de uso rápido (ajusta la ruta si hace falta)
    df = embeddings(
        "./cleaned_PHMRC_VAI_redacted_free_text.train.csv",
        method="bert",
    )
    print(df.head())
