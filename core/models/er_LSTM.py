import torch
from torch import nn

from core.models.blocks import MLP, fetch_input_dim

class ErLSTM(nn.Module):
    def __init__(self, config, hidden_dim=256, num_layers=1, bidirectional=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        # Fetch the correct input dimension based on backbone and modalities
        self.input_dimension = fetch_input_dim(config)

        # Initialize the LSTM encoder
        self.step_encoder = nn.LSTM(
            input_size=self.input_dimension,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Calculate the dimension after LSTM (double if bidirectional)
        lstm_output_dimension = hidden_dim * 2 if bidirectional else hidden_dim

        # Initialize the MLP decoder
        # We use the MLP block from blocks.py, which takes (input_size, hidden_size, output_size)
        self.decoder = MLP(lstm_output_dimension, 512, 1)

    def forward(self, input_data):
        # Check for NaNs in input and replace them with zero
        input_data = torch.nan_to_num(input_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # The collate_fn might flatten inputs to 2D [num_subsegments, features] 
        # For an LSTM, we need a 3D tensor: [batch_size, sequence_length, features]
        # We add a dummy batch dimension if it's missing (e.g., during step-level evaluation)
        is_2d = input_data.dim() == 2
        if is_2d:
            input_data = input_data.unsqueeze(0)  # Shape becomes [1, seq_len, input_dim]

        # Encode the sequence
        # lstm_out shape: [batch_size, seq_len, lstm_output_dimension]
        lstm_out, (hidden_state, cell_state) = self.step_encoder(input_data)
        
        # Pool the LSTM outputs across the sequence dimension (mean pooling)
        # This aggregates the temporal information of the entire step into a single vector
        pooled_output = torch.mean(lstm_out, dim=1)  # Shape: [batch_size, lstm_output_dimension]
        
        # Decode to get the error probability logit
        logits = self.decoder(pooled_output)  # Shape: [batch_size, 1]
        
        # If the input was originally 2D (sequence of sub-segments for a single step),
        # the evaluation pipeline might expect a logit for every sub-segment to do majority voting.
        # We expand the pooled logit to match the sequence length.
        if is_2d:
            seq_len = input_data.shape[1]
            logits = logits.expand(seq_len, -1)  # Shape: [seq_len, 1]

        return logits