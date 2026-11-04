"""
AgriSense AI - Module 4: Sequential Irrigation Models (RNN, LSTM, GRU, BiLSTM)
Defines architectures for time-series soil moisture forecasting.
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class BiLSTM(nn.Module):
        """
        Bidirectional LSTM regressor for soil moisture forecasting.
        Constructed by Member C.
        """
        def __init__(
            self,
            input_size=11,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            output_size=1
        ):
            super(BiLSTM, self).__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True
            )

            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size * 2, output_size)

        def forward(self, x):
            output, (h_n, c_n) = self.lstm(x)
            forward_hidden = h_n[-2]
            backward_hidden = h_n[-1]
            last_output = torch.cat((forward_hidden, backward_hidden), dim=1)
            last_output = self.dropout(last_output)
            return self.fc(last_output)


    class LSTM(nn.Module):
        """
        Unidirectional LSTM regressor (for Role B integration).
        """
        def __init__(
            self,
            input_size=11,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            output_size=1
        ):
            super(LSTM, self).__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False
            )

            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            output, (h_n, c_n) = self.lstm(x)
            last_output = self.dropout(h_n[-1])
            return self.fc(last_output)


    class GRU(nn.Module):
        """
        Gated Recurrent Unit regressor (for Role B integration).
        """
        def __init__(
            self,
            input_size=11,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            output_size=1
        ):
            super(GRU, self).__init__()

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            output, h_n = self.gru(x)
            last_output = self.dropout(h_n[-1])
            return self.fc(last_output)
else:
    class BiLSTM:
        pass
    class LSTM:
        pass
    class GRU:
        pass

