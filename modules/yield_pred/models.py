"""
AgriSense AI - Modules 1 & 2: Crop Yield Prediction MLP
Implements Multi-Layer Perceptron regressor with configurable:
- Activations (ReLU, Tanh, LeakyReLU)
- Regularization (Dropout, L2 Weight Decay)
- Weight Initializations (He/Kaiming, Xavier/Glorot, Random)
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CropYieldMLP(nn.Module):
        def __init__(
            self,
            input_dim=7,           # N, P, K, temperature, humidity, pH, rainfall
            hidden_dims=[64, 32, 16],
            output_dim=1,          # Predicted Yield (e.g., metric tons per hectare)
            activation="relu",     # "relu", "tanh", "leakyrelu"
            dropout_rate=0.2,      # Regularization ablation
            init_method="he"       # "he", "xavier", "random"
        ):
            super(CropYieldMLP, self).__init__()

            # Select activation function
            act_map = {
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "leakyrelu": nn.LeakyReLU(0.1)
            }
            self.act_fn = act_map.get(activation.lower(), nn.ReLU())

            layers = []
            prev_dim = input_dim

            for h_dim in hidden_dims:
                linear = nn.Linear(prev_dim, h_dim)
                self._apply_init(linear, init_method, activation)
                layers.append(linear)
                layers.append(self.act_fn)
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = h_dim

            # Output layer
            out_linear = nn.Linear(prev_dim, output_dim)
            self._apply_init(out_linear, init_method, activation)
            layers.append(out_linear)

            self.network = nn.Sequential(*layers)

        def _apply_init(self, layer, method, activation):
            if not isinstance(layer, nn.Linear):
                return
            m = method.lower()
            if m == "he":
                nonlin = "leaky_relu" if activation == "leakyrelu" else "relu"
                nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlin)
            elif m == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.normal_(layer.weight, mean=0.0, std=0.05)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        def forward(self, x):
            return self.network(x)
else:
    class CropYieldMLP:
        pass

