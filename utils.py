# utils.py

def get_complete_config(config):
    """
    Rellena un diccionario de configuración con valores por defecto.
    No hay relabelling aquí.
    """
    defaults = {
        "seed": 42,
        "pca_dim": 200,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 100,
        "batch_size": 32,
        "f1_mode": "weighted",

        # Vectorización
        "embedding_type": "bert",  # 'bert', 'tfidf', 'bow'
        "vectorizer_params": {},  # params específicos para TF-IDF/BOW

        # Datos
        "data_path": "./cleaned_PHMRC_VAI_redacted_free_text.train.csv",
        "text_col": "open_response",
        "id_col": "newid",
        "label_col": "gs_text34",
    }

    complete_config = {**defaults, **config}
    return complete_config
