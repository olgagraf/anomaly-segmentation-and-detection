# Adapted from:
# https://github.com/RobvanGastel/dinov3-finetune/
# Copyright (c) Rob van Gastel
# License: MIT
# Modified for this project.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_decoder import LinearClassifier
from .lora import LoRA


def _get_patch_size(model):
    patch_size = getattr(model, "patch_size", None)
    if patch_size is None:
        raise ValueError("Could not infer patch size from model. Expected model.patch_size.")
    return patch_size[0] if isinstance(patch_size, tuple) else int(patch_size)


class DINOV2EncoderLoRA(nn.Module):
    def __init__(
        self,
        encoder,
        r: int = 3,
        emb_dim: int = 768,
        n_classes: int = 10,
        use_lora: bool = False,
        img_dim: tuple[int, int] = (252, 252),
        parallel: bool = False,
        device: torch.device = None,
    ):
        super().__init__()
        self.parallel = parallel
        self.device = device if device else torch.device("cuda:0")

        encoder_module = encoder.module if self.parallel else encoder
        assert img_dim[0] % encoder_module.patch_size == 0, "Wrong input shape for patches"
        assert r > 0

        self.emb_dim = emb_dim
        self.img_dim = img_dim
        self.patch_size = int(encoder_module.patch_size)
        self.use_lora = use_lora

        self.encoder = encoder.to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder
        self.decoder = LinearClassifier(
            emb_dim,
            patch_h=int(img_dim[0] / encoder_module.patch_size),
            patch_w=int(img_dim[1] / encoder_module.patch_size),
            n_classes=n_classes,
        ).to(self.device)

        # Add LoRA layers to the encoder
        if self.use_lora:
            self.w_a = []
            self.w_b = []

            for block in encoder_module.blocks:
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r, self.device)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r, self.device)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                for layer in [w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v]:
                    layer.to(self.device)
                    layer.weight.data = layer.weight.data.to(self.device)
                    if layer.bias is not None:
                        layer.bias.data = layer.bias.data.to(self.device)

                block.attn.qkv = LoRA(
                    w_qkv_linear.to(self.device),
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    device=device,
                ).to(self.device)

            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int, device: torch.device):
        w_a = nn.Linear(dim, r, bias=False).to(device)
        w_b = nn.Linear(r, dim, bias=False).to(device)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_module = self.encoder.module if self.parallel else self.encoder
        x = x.to(self.device)
        feature = encoder_module.forward_features(x)
        patch_embeddings = feature["x_norm_patchtokens"]
        logits = self.decoder(patch_embeddings)
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits

    def save_parameters(self, filename: str) -> None:
        w_a, w_b = {}, {}
        if self.use_lora:
            w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
            w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}

        decoder_weights = self.decoder.state_dict()
        torch.save({**w_a, **w_b, **decoder_weights}, filename)

    def load_parameters(self, filename: str, map_location="cuda:0") -> None:
        state_dict = torch.load(filename, map_location=map_location)

        if self.use_lora:
            for i, w_A_linear in enumerate(self.w_a):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_A_linear.weight = nn.Parameter(saved_tensor.to(self.device))

            for i, w_B_linear in enumerate(self.w_b):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_B_linear.weight = nn.Parameter(saved_tensor.to(self.device))

        decoder_head_dict = self.decoder.state_dict()
        decoder_head_keys = [k for k in decoder_head_dict.keys()]
        decoder_state_dict = {k: state_dict[k] for k in decoder_head_keys}
        self.decoder.load_state_dict(decoder_state_dict)

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        encoder_module = self.encoder.module if self.parallel else self.encoder
        feature = encoder_module.forward_features(x)
        patch_embeddings = feature["x_norm_patchtokens"]
        return patch_embeddings
