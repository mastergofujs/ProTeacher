import torch
import math
from models.baseline_model import CNN
from models.conformer.conformer_encoder import ConformerEncoder
from models.conformer.downsampler import CNNLocalDownsampler
from models.transformer.encoder import Encoder as TransformerEncoder
import functools
from functools import reduce
from operator import mul
# import time
# from fvcore.nn import FlopCountAnalysis


class PromptEmbedding(torch.nn.Module):
    def __init__(self, prompt_config):
        super(PromptEmbedding, self).__init__()
        self.config = prompt_config
        self.num_tokens = self.config.num
        self.prompt_dim = self.config.dim
        self.drop_out = self.config.dropout
        self._get_prompt_embedding()
        
    def _get_prompt_embedding(self):
        # initiate prompt:
        # val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.prompt_embeddings = torch.nn.Parameter(torch.zeros(
            1, self.num_tokens, self.prompt_dim))
        # xavier_uniform initialization
        torch.nn.init.uniform_(self.prompt_embeddings.data)

        if self.config.DEEP:  # noqa
            total_d_layer = self.config["num_layers"]
            self.prompt_embeddings = torch.nn.Parameter(torch.zeros(
                total_d_layer, self.num_tokens, self.prompt_dim))
            # xavier_uniform initialization
            torch.nn.init.uniform_(self.prompt_embeddings.data)
        return self.prompt_embeddings
    
    
class SEDModel(torch.nn.Module):
    def __init__(
        self,
        n_class,
        cnn_kwargs=None,
        encoder_kwargs=None,
        prompt_kwargs=None,
        pooling="token",
        layer_init="pytorch",
    ):
        super(SEDModel, self).__init__()
        self.cnn_downsampler = CNNLocalDownsampler(n_in_channel=1, **cnn_kwargs)
        input_dim = self.cnn_downsampler.cnn.nb_filters[-1]
        adim = encoder_kwargs["adim"]
        self.pooling = pooling
        self.encoder = ConformerEncoder(input_dim, **encoder_kwargs)
        self.pred_head = torch.nn.Linear(adim, n_class)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, n_class)
            self.sigmoid = torch.sigmoid
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            # self.cls_token = torch.nn.Linear(1, input_dim)
            self.tag_token = torch.nn.Parameter(torch.zeros(1, 1, input_dim))
            self.prompt_tokens = PromptEmbedding(prompt_kwargs)

        self.reset_parameters(layer_init)

    def forward(self, x, mask=None, prompt_tuning=False):
        x = self.cnn(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        
        if self.pooling == "token":
            if prompt_tuning:
                x = torch.cat([self.tag_token, self.prompt_tokens, x], dim=1)
            else:
                x = torch.cat([self.tag_token, x], dim=1)
            
            # x = torch.cat([self.tag_token.expand(x.size(0), -1, -1), x], dim=1)
        
        x, _ = self.encoder(x, mask)

        if self.pooling == "attention":
            strong = self.pred_head(x)
            sof = self.dense(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (torch.sigmoid(strong) * sof).sum(1) / sof.sum(1)  # [bs, nclass]
            # Convert to logit to calculate loss with bcelosswithlogits
            weak = torch.log(weak / (1 - weak))
            
        elif self.pooling == "token":
            x = self.pred_head(x)
            weak = x[:, 0, :]
            if prompt_tuning:
                strong = x[:, self.prompt_tokens.num_tokens:, :]
            else:
                strong = x[:, 1:, :]
            
        return {"strong": strong, "weak": weak}

    def reset_parameters(self, initialization: str = "pytorch"):
        if initialization.lower() == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if initialization.lower() == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif initialization.lower() == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif initialization.lower() == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif initialization.lower() == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown initialization: {initialization}")
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()
                
    def get_masked_parameters(self, mask_param=None):
        if mask_param is not None:
            return self.parameters()