import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict 
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg19, VGG19_Weights

from models.encoder import Encoder
from models.decoder import Decoder


class MyResNet50(nn.Module):
    def __init__(self, resnet_model, output_layer):
        super(MyResNet50, self).__init__()
        self.output_layer = output_layer
        
        self._layers = []
        for l in list(resnet_model._modules.keys()):
            self._layers.append(l)
            if l == output_layer:
                break
        self.layers = OrderedDict(zip(self._layers,[getattr(resnet_model,l) for l in self._layers]))

    def _forward_impl(self, x):
        for l in self._layers:
            x = self.layers[l](x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    


# Define the LSTM encoder-decoder model
class BBCropsEncoderDecoder(nn.Module):
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
        feat_extractor = 'dino',
        fine_tuning = False, 
        normalization_layers = False
    ):
        super(BBCropsEncoderDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.target_len = target_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.normalization_layers = normalization_layers

        # ENCODER
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
        )
    
        if feat_extractor == 'dino': 
            self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
            visual_feat_size = 384
            self.fc2 = None

        elif feat_extractor == 'resnet':
            resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
            self.feature_extractor = MyResNet50(resnet_model, 'avgpool')
            visual_feat_size = int(hidden_size / 2)
            self.fc2 = nn.Linear(2048, visual_feat_size) # apply fc to reduce dimensionality
            
        elif feat_extractor == 'vgg':
            self.feature_extractor = vgg19(weights=VGG19_Weights.DEFAULT).to(device)
            visual_feat_size = int(hidden_size / 2)
            self.fc2 = nn.Linear(1000, visual_feat_size) # apply fc to reduce dimensionality
        
        if not fine_tuning:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size / 2))

        if self.normalization_layers:
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.layer_norm1 = nn.LayerNorm(hidden_size)
            self.layer_norm2 = nn.LayerNorm(visual_feat_size)
            self.fc3 = nn.Linear(hidden_size + visual_feat_size, hidden_size)

        # DECODER
        decoder_input_size = hidden_size if self.normalization_layers else int(hidden_size / 2) + visual_feat_size
        self.decoder = Decoder(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            batch_size=batch_size,
            num_layers=num_layers,
            device=device,
        )
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),         # converts the image to a tensor with values between 0 and 1
            transforms.Normalize(          # normalize to follow 0-centered imagenet pixel rgb distribution
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def normalize_crops(self, crops):
        crops = np.array(crops)
        norm_crops = []
        for c in crops:
            norm_crops.append(self.normalize(c))
            
        norm_crops = torch.stack(norm_crops)
        norm_crops = norm_crops.to(self.device)
        return norm_crops

    def forward(self, input, crops):
        '''
        : param input: input data (seq_length, input_size)
        : return outputs: torch containing predicted values
        '''
        # encode input
        _, encoder_hidden = self.encoder(input)
        encoder_out = encoder_hidden[1].reshape(self.batch_size, -1) # cell state
        
        # feature extraction
        encoded_features = self.feature_extractor(crops).squeeze()

        if self.fc2 is not None:
            encoded_features = self.relu(self.dropout(self.fc2(encoded_features)))

        # FC, activation function & dropout (& norm_layers if needed)
        if self.normalization_layers:
            out_fc1 = self.fc1(encoder_out)
            encoder_out = encoder_out + out_fc1
            encoder_out = self.layer_norm1(encoder_out)
            encoder_out = self.relu(self.dropout(encoder_out))

            encoded_features = self.layer_norm2(encoded_features)

            out = torch.cat((encoder_out, encoded_features), 1)
            out = self.relu(self.dropout(self.fc3(out)))

        else:
            encoder_out = self.relu(self.dropout(self.fc1(encoder_out)))
            out = torch.cat((encoder_out, encoded_features), 1)
        
        # initialize tensor for predictions
        outputs = torch.tensor([], device=self.device)

        # decode input
        decoder_input = out # shape: (batch_size, output_size)
        decoder_hidden = encoder_hidden
        
        # predict
        for t in range(self.target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs = torch.cat((outputs, decoder_output), 1)

        return outputs