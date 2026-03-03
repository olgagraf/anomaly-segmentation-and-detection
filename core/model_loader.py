import torch

from core.model.dino_v2 import DINOV2EncoderLoRA
from config import BACKBONES, EMBEDDING_DIMS, N_CLASSES


def load_dinov2_lora(
    size="base",
    r=3,
    use_lora=True,
    img_dim=(252, 252),
    parallel=True,
    dev="cuda:0",
    weights_path=None,
):
    """Load a DINOv2 encoder with LoRA adaptation.

    Returns:
        (model, device, emb_dim) tuple.
    """
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    emb_dim = EMBEDDING_DIMS[size]

    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=f"dinov2_{BACKBONES[size]}",
    )
    if parallel:
        encoder = torch.nn.DataParallel(encoder).to(device)

    dino_lora = DINOV2EncoderLoRA(
        encoder=encoder,
        r=r,
        emb_dim=emb_dim,
        img_dim=img_dim,
        n_classes=N_CLASSES,
        use_lora=use_lora,
        parallel=parallel,
        device=device,
    ).to(device)

    if weights_path is not None:
        dino_lora.load_parameters(weights_path)

    return dino_lora, device, emb_dim
