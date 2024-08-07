import torch

from models.baseline_model import CNN
from models.transformer.encoder import Encoder, PromptEncoder

class Transformer(torch.nn.Module):
    def __init__(
        self,
        n_class,
        cnn_kwargs=None,
        encoder_kwargs=None,
        pooling="token",
        layer_init="pytorch",
    ):
        super(Transformer, self).__init__()
        self.cnn = CNN(n_in_channel=1, **cnn_kwargs)
        input_dim = self.cnn.nb_filters[-1]
        # self.cnn_downsampler = CNNLocalDownsampler(n_in_channel=1, **cnn_kwargs)
        # input_dim = self.cnn_downsampler.cnn.nb_filters[-1]
        adim = encoder_kwargs["adim"]
        self.pooling = pooling
        self.encoder = PromptEncoder(input_dim, **encoder_kwargs)
        self.classifier = torch.nn.Linear(adim, n_class)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, n_class)
            self.sigmoid = torch.sigmoid
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            # self.linear_emb = torch.nn.Linear(1, input_dim)
            self.tag_token = torch.nn.Parameter(torch.zeros(1, 1, input_dim))

        self.reset_parameters(layer_init)

    def forward(self, x, mask=None, prompt_tuning=True):
        # input
        # x = x.repeat(1, 1, 3, 1)
        # x = self.cnn_downsampler(x)
        x = self.cnn(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        if self.pooling == "token":
            # tag_token = self.linear_emb(torch.ones(x.size(0), 1, 1).cuda())
            # x = torch.cat([self.tag_token, x], dim=1)
            x = torch.cat([self.tag_token.expand(x.size(0), -1, -1), x], dim=1)
        x, _ = self.encoder(x, prompt_tuning, mask)

        if self.pooling == "attention":
            strong = self.classifier(x)
            sof = self.dense(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (torch.sigmoid(strong) * sof).sum(1) / sof.sum(1)  # [bs, nclass]
            # Convert to logit to calculate loss with bcelosswithlogits
            weak = torch.log(weak / (1 - weak))
        elif self.pooling == "token":
            x = self.classifier(x)
            weak = x[:, 0, :]
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