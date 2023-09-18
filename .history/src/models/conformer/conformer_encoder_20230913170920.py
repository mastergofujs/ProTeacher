import torch
from models.conformer.conformer_block import ConformerBlock
from models.conformer.repeat import repeat
from models.transformer.embedding import PositionalEncoding
from models.conformer.masked_positional_encoding import MaskedPositionalEncoding

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
    
class ConformerEncoder(torch.nn.Module):
    # def __init__(self, n_stacks, d_model, d_ff, n_head, dropout):
    def __init__(
        self,
        idim: int,
        adim: int = 144,
        dropout_rate: float = 0.1,
        elayers: int = 3,
        eunits: int = 576,
        aheads: int = 4,
        kernel_size: int = 7,
    ):
        super(ConformerEncoder, self).__init__()
        assert adim % aheads == 0
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, adim),
            torch.nn.LayerNorm(adim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            PositionalEncoding(adim, dropout_rate),
        )
        self.conformer_blocks = repeat(elayers, lambda: ConformerBlock(adim, eunits, aheads, dropout_rate, kernel_size))

    def forward(self, x, mask=None):
        x = self.input_layer(x)
        x, mask = self.conformer_blocks(x, mask)
        return x, mask


class ConformerSeparator(torch.nn.Module):
    # def __init__(self, n_stacks, d_model, d_ff, n_head, dropout):
    def __init__(
        self,
        idim: int,
        adim: int = 144,
        dropout_rate: float = 0.1,
        elayers: int = 3,
        eunits: int = 576,
        aheads: int = 4,
        kernel_size: int = 7,
        n_class: int = 10
    ):
        super(ConformerSeparator, self).__init__()
        assert adim % aheads == 0
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, adim),
            torch.nn.LayerNorm(adim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            PositionalEncoding(adim, dropout_rate),
        )
        self.adim = adim
        self.n_class = n_class
        self.conformer_blocks = repeat(elayers, lambda: ConformerBlock(adim, eunits, aheads, dropout_rate, kernel_size))
        self.conv = torch.nn.Conv2d(adim, adim * n_class, kernel_size=2, stride=1, padding=1)
        self.norm = torch.nn.LayerNorm(adim * n_class)

    def forward(self, x, mask=None):
        x = self.input_layer(x)
        x, mask = self.conformer_blocks(x, mask)
        x = self.conv(x.permute(0, 2, 1).unsqueeze(-1))
        x = self.norm(x.squeeze(-1).permute(0, 2, 1))
        b, t, d = x.size()
        x = torch.relu(x.view(b, t, self.adim, self.n_class).contiguous().permute(0, 3, 1, 2))
        return x, mask

class ConformerMaskedEncoder(torch.nn.Module):
    # def __init__(self, n_stacks, d_model, d_ff, n_head, dropout):
    def __init__(
        self,
        idim: int,
        adim: int = 144,
        dropout_rate: float = 0.1,
        elayers: int = 3,
        eunits: int = 576,
        aheads: int = 4,
        kernel_size: int = 7,
    ):
        super(ConformerMaskedEncoder, self).__init__()
        assert adim % aheads == 0
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, adim),
            torch.nn.LayerNorm(adim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU())
        self.pe = MaskedPositionalEncoding(adim, dropout_rate)
        self.conformer_blocks = repeat(elayers, lambda: ConformerBlock(adim, eunits, aheads, dropout_rate, kernel_size))

    def forward(self, x, inds, cls_token=True, unmasked_only=False):
        masked_id, unmasked_id = inds[0], inds[1]
        if cls_token:
            inds_all = torch.cat([torch.zeros(1, dtype=torch.int64), masked_id + 1, unmasked_id + 1])
        else:
            inds_all = torch.cat([masked_id, unmasked_id])
        b, fn, d = x.shape
        x_u = torch.zeros((b, len(inds_all), d)).cuda()
        if not unmasked_only:
            x_u = x
            inds_all = inds_all.sort()[0]
        else: 
            if cls_token:
                unmasked_ = torch.cat([torch.zeros(1, dtype=torch.int64), unmasked_id + 1])
            else:
                unmasked_ = unmasked_id
            x_u[:, unmasked_, :] = x
            x_u = x_u[:, unmasked_.sort()[0], :]
            inds_all = unmasked_.sort()[0]
        x_u = self.input_layer(x_u)
        x_u = self.pe(x_u, inds_all)
        x, _ = self.conformer_blocks(x_u, None)
        return x, inds
    
class ConformerPromptedEncoder(torch.nn.Module):
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
        prompt_layers: int = 1
    ):
        super(ConformerPromptedEncoder, self).__init__()
        assert adim % aheads == 0
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, adim),
            torch.nn.LayerNorm(adim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            PositionalEncoding(adim, dropout_rate),
        )
        self.conformer_blocks = torch.nn.ModuleList()
        for e in range(elayers):
            self.conformer_blocks.append(ConformerBlock(adim, eunits, aheads, dropout_rate, kernel_size))        
        self.prompt_embedding = PromptEmbedding(idim, prompt_nums, prompt_layers, deep=False)
        
    def forward(self, x, mask=None):
        x = self.input_layer(x)
        x, mask = self.conformer_blocks(x, mask)
        return x, mask