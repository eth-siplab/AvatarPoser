import torch
import torch.nn as nn
from IPython import embed
import math


nn.Module.dump_patches = True


class AvatarPoser(nn.Module):
    def __init__(self, num_layer, input_dim, output_dim, hidden_dim, latent_dim, device):
        super(AvatarPoser, self).__init__()

        self.device = device

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.trans_en = nn.Linear(input_dim,self.hidden_dim)


        self.fc1a = nn.Linear(self.hidden_dim, 256)
        self.fc1b = nn.Linear(256, 6)

        self.fc2a = nn.Linear(self.hidden_dim, 256)
        self.fc2b = nn.Linear(256, 126)
        self.relu = nn.ReLU()
        self.init_weights()


        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1a.weight)
        nn.init.kaiming_normal_(self.fc1b.weight)
        nn.init.kaiming_normal_(self.fc2a.weight)
        nn.init.kaiming_normal_(self.fc2b.weight)

    def forward(self, input_tensor):

#        embed()
        out = self.trans_en(input_tensor)
        src = out.permute(1,0,2)
        out = self.transformer_encoder(src)
        out = out.permute(1,0,2)[:, -1]

        out1 = self.fc1b(self.relu(self.fc1a(out)))
        out2 = self.fc2b(self.relu(self.fc2a(out)))

        output = torch.cat([out1,out2],dim=-1)

        return output