# neural_networks/base_model.py
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Modelo base de red neuronal para clasificación multiclass.

    Arquitectura sencilla:
      - Linear(input_dim -> hidden_dim)
      - ReLU
      - Dropout
      - Linear(hidden_dim -> output_dim)
    """

    def __init__(self, input_dim, hidden_dim=None, output_dim=2, dropout_p=0.4):
        super().__init__()

        if hidden_dim is None:
            # Regla simple: hidden_dim ≈ 3/4 de input_dim
            hidden_dim = max(2, int(input_dim * 0.75))

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Inicialización Xavier para capas lineales.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
