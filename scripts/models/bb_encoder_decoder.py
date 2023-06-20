import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder

# Define the LSTM encoder-decoder model
class BBEncoderDecoder(nn.Module):
    '''train LSTM encoder-decoder and make predictions'''

    def __init__(
        self,
        input_size,
        output_size,
        batch_size,
        target_len,
        hidden_size,
        num_layers,
        device,
    ):
        super(BBEncoderDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.target_len = target_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
        )
        self.decoder = Decoder(
            input_size=output_size,
            hidden_size=hidden_size,
            output_size=output_size,
            batch_size=batch_size,
            num_layers=num_layers,
            device=device,
        )

    def forward(self, input):
        '''
        : param input: input data (seq_length, input_size)
        : return outputs: torch containing predicted values; prediction done recursively
        '''
        # encode input
        encoder_output, encoder_hidden = self.encoder(input)

        # initialize tensor for predictions
        outputs = torch.tensor([]).to(self.device)

        # we initialize the decoder with the last observed [∆x, ∆y, ∆w, ∆h] of each input sequence
        decoder_input = input[:, -1, :][:, 4:] 
        decoder_hidden = encoder_hidden

        # predict recursively
        for t in range(self.target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs = torch.cat((outputs, decoder_output), 1)
            decoder_input = decoder_output

        return outputs