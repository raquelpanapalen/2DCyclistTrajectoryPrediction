import torch.nn as nn

# Define the decoder module
class Decoder(nn.Module):
    '''Decodes hidden state output by encoder'''

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        batch_size,
        device,
        num_layers=1,
        drop_prob=0.5,
    ):
        '''
        param input_size:     the number of features in the input X
        param hidden_size:    the number of features in the hidden state h
        param num_layers:     number of recurrent layers (i.e., 2 means there are 2 stacked LSTMs)
        '''
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        # define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # define fully-connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, encoder_hidden_states):
        '''
        input: (batch_size, input_size) --> (batch_size, seq_length = 1, input_size)

        return:
          - output: (batch_size, seq_length = 1, output_size) --> gives all the hidden states in the sequence
          - hidden: (num_layers, batch_size, hidden_size) --> gives the hidden state and the cell state for the last element in the sequence
        '''
        input = input.unsqueeze(0).view(self.batch_size, 1, self.input_size)
        lstm_out, self.hidden = self.lstm(input, encoder_hidden_states)
        out = self.dropout(lstm_out)
        output = self.fc(out)

        return output, self.hidden