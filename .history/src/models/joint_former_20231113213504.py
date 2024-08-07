import torch
from models.conformer.conformer_encoder1 import ConformerEncoder, ConformerMaskedEncoder
from models.transformer.encoder import Encoder as TransformerEncoder
from models.conformer.downsampler import CNNLocalDownsampler, ConformerDownsamplerBlock
from models.conformer.conformer_decoder import ConformerMaskedDecoder, MaskedLinearDecoder

class SEDModel(torch.nn.Module):
    def __init__(
        self,
        n_class,
        cnn_kwargs=None,
        encoder_kwargs=None,
        decoder_kwargs=None,
        mask_kwargs=None,
        encoder_type="Conformer",
        pooling="token",
        layer_init="pytorch",
    ):
        super(SEDModel, self).__init__()
        self.mask_size = mask_kwargs['mask_size']
        self.mask_ratio = mask_kwargs['mask_ratio']
        self.cnn_downsampler = CNNLocalDownsampler(n_in_channel=1, **cnn_kwargs)
        input_dim = self.cnn_downsampler.cnn.nb_filters[-1]
        adim = encoder_kwargs["adim"]
        self.pooling = pooling

        if encoder_type == "Transformer":
            self.encoder = TransformerEncoder(input_dim, **encoder_kwargs)
        elif encoder_type == "Conformer":
            self.encoder_masked = ConformerMaskedEncoder(input_dim, **encoder_kwargs)
        else:
            raise ValueError("Choose encoder_type in ['Transformer', 'Conformer']")

        self.classifier = torch.nn.Linear(adim, n_class)

        if self.pooling == "attention":
            self.dense = torch.nn.Linear(adim, n_class)
            self.sigmoid = torch.sigmoid
            self.softmax = torch.nn.Softmax(dim=-1)

        elif self.pooling == "token":
            self.tag_token = torch.nn.Parameter(torch.zeros(1, 1, input_dim))
            self.tag_token = torch.nn.init.xavier_normal_(self.tag_token)
        
        self.decoder = ConformerMaskedDecoder(**decoder_kwargs)
        self._recon_loss = torch.nn.MSELoss()
        self.reset_parameters(layer_init)

    def forward(self, input, prompt_tuning=True):
        embs = input
        x = self.cnn_downsampler(input)
        x_m, masked_inds, unmasked_inds = self.cnn_downsampler(
            input, mask=True, 
            mask_size=self.mask_size, 
            mask_ratio=self.mask_ratio)
        if self.pooling == "token":
            x_m = torch.cat([self.tag_token.expand(input.size(0), -1, -1), x_m], dim=1)
            x = torch.cat([self.tag_token.expand(input.size(0), -1, -1), x], dim=1)
            cls_token = True
        else:
            cls_token = False
        x, _ = self.encoder_masked(x, inds=(masked_inds, unmasked_inds), cls_token=cls_token)
        x_m, _ = self.encoder_masked(x_m, inds=(masked_inds, unmasked_inds), 
                                     unmasked_only=True, cls_token=cls_token)
        dec = self.decoder(x_m, inds=(masked_inds, unmasked_inds), cls_token=cls_token)

        recon_loss = self._recon_loss(dec, input.squeeze(1))
        #recon_loss = self._recon_loss(dec[:, masked_inds], input.squeeze(1)[:, masked_inds]) #mask only
        x = self.classifier(x)
        if self.pooling == "token": 
            weak = x[:, 0, :]
            strong = x[:, 1:, :]
        elif self.pooling == "hipool":
            strong = torch.sigmoid(x)
            weak = self.hipool(strong)
        return {"strong": strong, "weak": weak, 'recon_loss': recon_loss}

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
    
