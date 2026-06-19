import torch
import random
import os
import numpy as np
from torch import nn
from typing import List, Dict

# -------------------------
# Utils
# -------------------------
def seed_torch(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(name: str, params, lr: float):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")

def get_model_path(decoder_type: str) -> str:
    paths = {
        "GPT2": "openai-community/gpt2",
        "GPT2-medium": "openai-community/gpt2-medium",
    }
    if decoder_type in paths:
        return paths[decoder_type]
    raise ValueError(f"Unsupported decoder_type: {decoder_type}")

def calculate_accuracies_shared_dual(coarse_preds, fine_preds, coarse_tgt, fine_tgt, tokenizer):
    tgt_c = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in coarse_tgt]
    tgt_f = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in fine_tgt]

    c_corr = sum(1 for p, t in zip(coarse_preds, tgt_c) if str(p).strip() == str(t).strip())
    f_corr = sum(1 for p, t in zip(fine_preds, tgt_f) if str(p).strip() == str(t).strip())
    return c_corr, f_corr, len(coarse_tgt)

def _resize_linear_out_features(old: nn.Linear, new_out: int) -> nn.Linear:
    device = old.weight.device
    dtype = old.weight.dtype
    in_features = old.in_features

    new_layer = nn.Linear(in_features, new_out, bias=(old.bias is not None)).to(device=device, dtype=dtype)

    with torch.no_grad():
        k = min(old.out_features, new_out)
        new_layer.weight[:k].copy_(old.weight[:k])
        if old.bias is not None and new_layer.bias is not None:
            new_layer.bias[:k].copy_(old.bias[:k])
    return new_layer

def _patch_unembedding_layer(obj, attr_name, vocab_len):
    if hasattr(obj, attr_name):
        layer = getattr(obj, attr_name)
        if layer.out_features != vocab_len:
            setattr(obj, attr_name, _resize_linear_out_features(layer, vocab_len))

def patch_unembedding_to_tokenizer(model, tokenizer):
    vocab_len = len(tokenizer)
    impl = getattr(model, "impl", model)

    if hasattr(impl, "decoder") and hasattr(impl.decoder, "coarse_unembedding"):
        _patch_unembedding_layer(impl.decoder, "coarse_unembedding", vocab_len)
        _patch_unembedding_layer(impl.decoder, "fine_unembedding", vocab_len)

    if hasattr(impl, "coarse_branch"):
        _patch_unembedding_layer(impl.coarse_branch, "unembedding", vocab_len)

    if hasattr(impl, "fine_branch"):
        _patch_unembedding_layer(impl.fine_branch, "unembedding", vocab_len)

    if hasattr(impl, "branch"):
        _patch_unembedding_layer(impl.branch, "unembedding", vocab_len)


@torch.no_grad()
def mean_pool_label_embeddings(tokenizer, wte, labels: List[str], device: torch.device) -> torch.Tensor:
    tok = tokenizer(labels, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
    emb = wte(tok.input_ids)  # (L, T, H)
    mask = tok.attention_mask.unsqueeze(-1).float()
    pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled  # (L, H)

def build_text_bank_for_losses(
    model,
    tokenizer,
    device: torch.device,
    all_coarse_labels: List[str],
    all_fine_labels: List[str],
) -> Dict[str, torch.Tensor]:
    impl = getattr(model, "impl", model)

    projector = getattr(impl, "text_projector", None)
    if projector is None:
        projector = nn.Identity().to(device)

    wte_c = impl.decoder.shared_trunk.wte
    wte_f = impl.decoder.shared_trunk.wte

    raw_c = mean_pool_label_embeddings(tokenizer, wte_c, all_coarse_labels, device)
    raw_f = mean_pool_label_embeddings(tokenizer, wte_f, all_fine_labels, device)

    proj_c = projector(raw_c)
    proj_f = projector(raw_f)
    proj_c = torch.nn.functional.normalize(proj_c, dim=-1)
    proj_f = torch.nn.functional.normalize(proj_f, dim=-1)

    return {"raw_coarse": raw_c, "raw_fine": raw_f, "proj_coarse": proj_c, "proj_fine": proj_f}
