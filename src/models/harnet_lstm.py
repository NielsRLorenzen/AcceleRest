import torch
import torch.nn as nn
import torch.nn.functional as F

class HARNetLSTM(nn.Module):
    def __init__(
            self, 
            harnet: nn.Module,
            num_lstm_layers: int,
            dropout: float,
            patch_size: int,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.feature_extractor = harnet.feature_extractor
        self.classifier = harnet.classifier
        lstm_input_size = self.classifier.linear1.in_features
        # Divide by 2 because of bidirectional LSTM
        lstm_hidden_size = lstm_input_size // 2
        lstm_dropout = dropout if num_lstm_layers > 1 else 0
        self.lstm = nn.LSTM(
            lstm_input_size,
            lstm_hidden_size,
            num_lstm_layers,
            dropout = lstm_dropout,
            batch_first = True,
            bidirectional = True,
        )
        self.head_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (batch_size, channels, num_patches, patch_size)
        x  = x.unfold(-1, self.patch_size, self.patch_size)

        # Shape (batch_size, num_patches, channels, patch_size)
        x = x.permute(0, 2, 1, 3)
        batch_size, num_patches = x.shape[:2]

        # Shape (batch_size * num_patches, channels, patch_size)
        x = x.flatten(start_dim=0, end_dim=1)

        # Shape (batch_size * num_patches, lstm_input_size)
        x = self.feature_extractor(x).squeeze(-1)

        # Shape (batch_size, num_patches, lstm_input_size)
        x = x.unflatten(0, (batch_size, num_patches))
        
        # Shape (batch_size, num_patches, lstm_hidden_size * 2)
        x, _ = self.lstm(x)
        
        # Shape (batch_size, num_patches, num_classes)
        y_hat = self.classifier(self.head_dropout(x))

        # Permute to shape expected by loss function
        y_hat = y_hat.permute(0,2,1) # (batch_size, num_classes, num_patches)
        return y_hat