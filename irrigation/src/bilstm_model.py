import torch
import torch.nn as nn


class BiLSTM(nn.Module):

    def __init__(
        self,
        input_size=11,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=3
    ):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):

        output, (h_n, c_n) = self.lstm(x)

        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]

        last_output = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        last_output = self.dropout(last_output)

        prediction = self.fc(last_output)

        return prediction


if __name__ == "__main__":

    model = BiLSTM(
        input_size=11,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=1
    )

    sample = torch.randn(32, 24, 11)

    result = model(sample)

    print("BiLSTM model created successfully!")
    print("Input shape:", sample.shape)
    print("Output shape:", result.shape)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))