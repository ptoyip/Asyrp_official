import torch
import numpy as np
import math
import torchvision
from nystrom_attention import NystromAttention
from diffusion_latent import Asyrp
from einops import rearrange


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Nystrom_MSA(torch.nn.Module):
    def __init__(self, norm_layer=torch.nn.LayerNorm, dim=512) -> None:
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        out, attn = self.attn(self.norm(x), return_attn=True)
        x = x + out
        return x, out, attn


class SD(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.complete_swin = torchvision.models.swin_t(weights="DEFAULT")
        # Swin Transformer input: (256, 256)
        self.extractor = torch.nn.Sequential(*list(self.complete_swin.children())[:-5])
        self.nystrom_msa = Nystrom_MSA(norm_layer=torch.nn.LayerNorm, dim=768)
        self._fc1 = torch.nn.Linear(in_features=64 * 768, out_features=1)
        self.asyrp = Asyrp()  #! Add Asyrp args and configs

    def forward(self, x):
        # x: b, w = 256, h = 256, c = 3
        feat = self.extractor(x)

        # feat = (b, w/32, h/32, 8*C = 768)
        b, w, h, c = feat.shape
        feat_vector = feat.flatten(start_dim=1, end_dim=2)

        # msa_feat = (b, (w, h), 8C) = (1, 64, 768), msa_attn = (b, head, w, h)
        msa_feat_vector, msa_out, attn = self.nystrom_msa(feat_vector)
        # print(msa_out.shape)
        delta_h = self._fc1(msa_out.flatten())

        # feature-wise asyrp ()
        msa_feat = rearrange(msa_feat_vector, "b (w h) c -> b w h c", w=w)
        msa_feat_prime = self.asyrp(x, delta_h)
        # msa_feat_prime = 1
        # ? To use msa_out or attn from Nystromer?
        return msa_feat, msa_feat_prime, msa_out, feat


class SD_MIL(torch.nn.Module):
    def __init__(self, subclass) -> None:
        super().__init__()
        self.subclass = subclass
        self.sd_block = SD()
        self.asyrp = Asyrp()  #! Add Asyrp args in merged level and configs
        self.pos_encoder = PositionalEncoding(dim=768)
        self.nystrom_msa1 = Nystrom_MSA(
            norm_layer=torch.nn.LayerNorm, dim=768
        )  # ? Check dim
        self.nystrom_msa2 = Nystrom_MSA(
            norm_layer=torch.nn.LayerNorm, dim=1
        )  # ? Check dim
        self.mlp = torch.nn.Linear(
            in_features=16 * 16 * 786, out_features=self.subclass
        )

    def forward(self, x):
        low_feat, low_feat_prime, low_out, swin_feat = self.sd_block(x)
        b, w, h, c = x.shape

        # * ~~~~~~~~~~~~~~~ LHS ~~~~~~~~~~~~~~~
        pos_encoding = self.pos_encoder(low_feat)
        # ? Do we need to make one more positional encoding for low_feat' or we just use the same?
        low_pos_feat = low_feat + pos_encoding
        low_pos_feat_prime = low_feat_prime + pos_encoding

        # Nystrom MSA [b, 64*2, 768]
        # ? By concat dim0, we try to find attention relationship between different instances
        # ? By concat dim1, we try to find attention relationship between orginal feats and generated feats
        merge_feat = torch.concat([low_pos_feat, low_pos_feat_prime], dim=0)
        merge_msa_feat, merge_msa_out, merge_attn = self.nystrom_msa1(merge_feat)
        low_msa_feat_vec = merge_msa_feat[:, : (w * h) // 2, :]
        low_msa_feat_prime_vec = merge_msa_feat[:, (w * h) // 2 :, :]
        low_msa_feat = rearrange(low_msa_feat_vec, "b (w h) c -> b w h c", w=w)
        low_msa_feat_prime = rearrange(
            low_msa_feat_prime_vec, "b (w h) c -> b w h c", w=w
        )

        # Feature Merge
        high_feat = self.instance_merge(
            low_feat=low_msa_feat, low_feat_prime=low_msa_feat_prime
        )

        # * ~~~~~~~~~~~~~~~ RHS ~~~~~~~~~~~~~~~
        # TODO: Change the naive high feat merge into linear transform with attention weighting
        naive_delta_h = None
        high_naive_feat = self.instance_merge(low_feat)
        high_naive_feat_prime = self.asyrp(high_naive_feat, naive_delta_h)

        # * ~~~~~~~~~~~~~~~ FINAL ~~~~~~~~~~~~~~~
        high_pos_encoding = self.pos_encoder(high_feat)
        high_pos_feat = high_feat + high_pos_encoding
        high_naive_pos_feat_prime = high_naive_feat_prime + high_pos_encoding

        merge_high_feat = torch.concat(
            [high_pos_feat, high_naive_pos_feat_prime], dim=0
        )
        merge_high_feat_vector = merge_high_feat.flatten(start_dim=1, end_dim=2)
        final_feat, final_out, final_attn = self.nystrom_msa2(merge_high_feat)
        logits = self.mlp(final_feat.flatten())
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = torch.functional.softmax(logits, dim=1)
        results_dict = {"logits": logits, "Y_prob": Y_prob, "Y_hat": Y_hat}
        return results_dict

    def instance_merge(self, low_feat, low_feat_prime=None):
        b, w, h, c = low_feat.shape
        if low_feat_prime is not None:
            if not b % 4:
                high_feat = rearrange(
                    low_feat, "(b b1 b2) w h c -> b (w b1) (h b2) c", b1=2, b2=2
                )
                high_prime_feat = rearrange(
                    low_feat_prime, "(b b1 b2) w h c -> b (w b1) (h b2) c", b1=2, b2=2
                )
                high_feat = torch.concat([high_feat, high_prime_feat], dim=0)
            elif not b % 2:
                high_feat = rearrange(low_feat, "(b b1) w h c -> b (w b1) h c", b1=2)
                high_prime_feat = rearrange(
                    low_feat_prime, "(b b1) w h c -> b (w b1) h c", b1=2
                )
                high_feat = torch.concat([high_feat, high_prime_feat], dim=2)
            else:
                #! Drop the lowest attention score feature?
                raise ValueError(
                    "Number of low level instance must be even to perform feature merge"
                )
        else:
            if not b % 4:
                high_feat = rearrange(
                    low_feat, "(b b1 b2) w h c -> b (w b1) (h b2) c", b1=2, b2=2
                )
            else:
                #! Drop the lowest attention score feature?
                raise ValueError(
                    "Number of low level instance must be multiple of 4 to perform feature merge"
                )
        return high_feat
