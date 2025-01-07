import torch
import torch.nn as nn
from tools.attn import ProbAttention, AttentionLayer
from tools.embed import DataEmbedding
from tools.encoder import Encoder, EncoderLayer, ConvLayer
from tools.ODconv import ODConv1d
import math

class TimeFeatureBlock(nn.Module):
    def __init__(self,args, env_days):
        super(TimeFeatureBlock, self).__init__()

        self.args = args
        self.output_attention = False

        # Encoding
        self.enc_embedding = DataEmbedding(args.enc_in, d_model=128, embed_type='timeF', freq='d', dropout=0.05)
        # Attention
        Attn = ProbAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor=5, attention_dropout=0.05, output_attention=self.output_attention), 
                                d_model=126, n_heads=6, mix=False),
                    d_model=126,
                    d_ff=2048,
                    dropout=0.05,
                    activation='gelu'
                ) for l in range(2)
            ],
            [
                ConvLayer(126) for l in range(1)
            ] ,
            norm_layer=torch.nn.LayerNorm(126)
        )
       
        self.projection = nn.Linear(126, args.c_out, bias=True)

        self.fc = nn.Linear(math.ceil(env_days/2), 1)

        self.fc2 = nn.Sequential(
            nn.Linear(125, 76),
            nn.ReLU(),
            nn.Linear(76, 38)
        )

        self.conv_env = nn.Sequential(
            nn.Conv1d(in_channels=75, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.ODconv = ODConv1d(env_days, env_days, 3) 
        


    def forward(self, x_enc, x_mark_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.ODconv(enc_out)

        enc_out, self.attns = self.encoder(enc_out, attn_mask=enc_self_mask)
    
        enc_out = enc_out.permute(0,2,1)
        enc_out = self.fc(enc_out)
        enc_out = enc_out.permute(0,2,1)

        return enc_out 