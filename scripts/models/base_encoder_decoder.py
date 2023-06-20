import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder

# Define the LSTM encoder-decoder model
class BaseEncoderDecoder(nn.Module):
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
        super(BaseEncoderDecoder, self).__init__()
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

    def forward(self, input, target_len):
        '''
        : param input: input data (seq_length, input_size)
        : param target_len: number of target values to predict
        : return np_outputs: np.array containing predicted values; prediction done recursively
        '''
        # encode input
        encoder_output, encoder_hidden = self.encoder(input)

        # initialize tensor for predictions
        outputs = torch.tensor([]).to(self.device)

        # decode input
        decoder_input = input[:, -1, :]
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs = torch.cat((outputs, decoder_output), 1)
            decoder_input = decoder_output

        return outputs
