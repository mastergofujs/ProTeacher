import torch
from torch import nn
from models.transformer.attention import MultiHeadedAttention
from models.transformer.embedding import PositionalEncoding
from models.transformer.encoder_layer import EncoderLayer
from models.transformer.layer_norm import LayerNorm
from models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from models.transformer.repeat import repeat
from models.transformer.subsampling import Conv2dNoSubsampling, Conv2dSubsampling

# Reference: https://github.com/espnet/espnet/tree/master/espnet/nets/pytorch_backend/transformer
class PromptEmbedding():
    def __init__(self, dim, nums, layers=1, deep=False):
        super(PromptEmbedding, self).__init__()
        self.num_tokens = nums
        self.prompt_dim = dim
        self.deep = deep
        self.layers = layers
        
    def _get_prompt_embedding(self):
        # initiate prompt:
        # val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        prompt_embeddings = torch.nn.Parameter(torch.zeros(
            1, self.num_tokens, self.prompt_dim))
        # xavier_uniform initialization
        torch.nn.init.uniform_(prompt_embeddings.data)

        if self.deep: 
            prompt_embeddings = torch.nn.Parameter(torch.zeros(
                self.layers, self.num_tokens, self.prompt_dim))
            # xavier_uniform initialization
            torch.nn.init.uniform_(prompt_embeddings.data)
        return prompt_embeddings

class Encoder(torch.nn.Module):
    """Encoder module
    :param int idim: input dim
    :param argparse.Namespace args: experiment config
    """

    def __init__(self, idim, args, pos_enc=True):
        super(Encoder, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, args['adim']),
            torch.nn.LayerNorm(args['adim']),
            torch.nn.Dropout(args['dropout_rate']),
            torch.nn.ReLU(),
            PositionalEncoding(args['adim'], args['dropout_rate']),
        )
        self.encoders = repeat(
            args['elayers'],
            lambda: EncoderLayer(
                args['adim'],
                MultiHeadedAttention(args['aheads'], args['adim'], 0.2),
                PositionwiseFeedForward(args['adim'], args['eunits'], 0.2),
                args['dropout_rate'],
                False,
            ),
        )
        self.norm = LayerNorm(args['adim'])

    def forward(self, x, mask=None):
        """Embed positions in tensor
        :param torch.Tensor x: input tensor
        :param torch.Tensor mask: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        x = self.input_layer(x)
        x, mask = self.encoders(x, mask)
        return self.norm(x), mask
    
class PromptEncoder(nn.Module):
    def __init__(
        self,
        idim: int,
        adim: int = 144,
        dropout_rate: float = 0.1,
        elayers: int = 3,
        eunits: int = 576,
        aheads: int = 4,
        kernel_size: int = 7,
        prompt_nums: int = 10,
        prompt_layers: int = 3,
    ):
        super(PromptEncoder, self).__init__()
        assert adim % aheads == 0
        self.elayer = elayers
        self.player = prompt_layers
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, adim),
            torch.nn.LayerNorm(adim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU()
        )
        self.pe = MaskedPositionalEncoding(adim, dropout_rate)
        self.conformer_blocks = torch.nn.ModuleList()
        for e in range(elayers):
            self.conformer_blocks.append(ConformerBlock(adim, eunits, aheads, dropout_rate, kernel_size))        
        self.prompt_tokens = PromptEmbedding(idim, prompt_nums, self.player, deep=True)._get_prompt_embedding()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(idim, adim),
            torch.nn.LayerNorm(adim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU()
            )
        
    def forward(self, x, mask=None, prompt_tuning=True):
        seq_len = x.size(1) - 1
        if prompt_tuning:
            x = self.input_layer(torch.cat(
                [x[:, 0:1, :], self.prompt_tokens[0, :, :].expand(x.size(0), -1, -1), x[:, 1:, :]],
                dim=1
            ))
            ind = torch.arange(0, x.size(1))
            x = self.pe(x, ind)
        else:
            x = self.input_layer(x)
            ind = torch.arange(self.prompt_tokens.size(1), self.prompt_tokens.size(1) + seq_len + 1)
            ind[0] = 0
            x = self.pe(x, ind.squeeze())
        for e in range(self.elayer):
            # [cls token, prompt token, data]
            if prompt_tuning:
                if e < self.player:
                    if e > 0 :
                        x[:, 1: 1 + self.prompt_tokens.size(1), :] = self.fc(self.prompt_tokens[e, :, :]).expand(x.size(0), -1, -1)
            x, mask = self.conformer_blocks[e](x, mask)
        x = torch.cat([x[:, 0:1], x[:, - seq_len:]], dim=1)
        return x, mask