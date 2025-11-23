# evaluacion.py
import torch
from torchmetrics.classification import F1Score


def fscore(num_clases, y_true, y_pred, le=None, mode="weighted"):
    """
    Calcula F1-score global y por clase, sin relabelling.

    Parámetros:
      - num_clases: número de clases (entero)
      - y_true: tensor 1D con etiquetas verdaderas (enteros)
      - y_pred: tensor 1D con predicciones (enteros)
      - le: LabelEncoder opcional, para mapear índices a nombres de clase
      - mode: 'weighted', 'macro', etc. para el F1 global

    Devuelve:
      {
        "f1_global": float,
        "f1_por_clase": {nombre_clase: f1, ...}
      }
    """
    # Aseguramos CPU
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    # F1 global
    f1_global_metric = F1Score(task="multiclass", num_classes=num_clases, average=mode)
    f1_global_val = f1_global_metric(y_pred, y_true).item()

    # F1 por clase
    f1_per_class_metric = F1Score(task="multiclass", num_classes=num_clases, average=None)
    f1_vals = f1_per_class_metric(y_pred, y_true).cpu().numpy().tolist()

    # Construimos diccionario por clase
    if le is not None:
        nombres_clase = list(le.classes_)
    else:
        nombres_clase = [f"class_{i}" for i in range(num_clases)]

    f1_por_clase = {nombres_clase[i]: f1_vals[i] for i in range(len(f1_vals))}

    return {"f1_global": f1_global_val, "f1_por_clase": f1_por_clase}
