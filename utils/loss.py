from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
# --------------------------------------------
# 1) HIER: TextHierarchicalLoss
# --------------------------------------------
class TextHierarchicalLoss(nn.Module):
    def __init__(self, fine_labels, coarse_labels, hierarchy_map,
                 tau=0.07, reg=True,
                 dist_lambda_scale=0.5,
                 margin=0.5):
        super().__init__()
        self.tau = float(tau)
        self.reg = bool(reg)

        self.dist_lambda_scale = float(dist_lambda_scale)
        self.margin = float(margin)

        self.num_fine = len(fine_labels)
        self.num_coarse = len(coarse_labels)

        fine_to_idx = {name: i for i, name in enumerate(fine_labels)}
        coarse_to_idx = {name: i for i, name in enumerate(coarse_labels)}

        parent_idx = torch.full((self.num_fine,), -1, dtype=torch.long)
        for fine_name, coarse_name in hierarchy_map.items():
            if fine_name in fine_to_idx and coarse_name in coarse_to_idx:
                parent_idx[fine_to_idx[fine_name]] = coarse_to_idx[coarse_name]
        self.register_buffer("fine_to_parent_idx", parent_idx, persistent=False)

    @staticmethod
    def _log_weighted_sumexp(lse: torch.Tensor, weight: float = 1.0, device=None) -> torch.Tensor:
        if device is None:
            device = lse.device
        if weight > 0.0:
            return torch.log(torch.tensor(weight, device=device, dtype=lse.dtype)) + lse
        return lse

    def _apply_penalty(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def forward(
        self,
        fine_embeddings: torch.Tensor,          
        coarse_embeddings: torch.Tensor,       
        raw_fine_embeddings: torch.Tensor = None,  
    ):
        device = fine_embeddings.device
        parent = self.fine_to_parent_idx.to(device)

        f_norm_main = F.normalize(fine_embeddings, dim=-1)
        c_norm = F.normalize(coarse_embeddings, dim=-1)

        # Similarities (main)
        s_cf = c_norm @ f_norm_main.T
        s_ff_main = f_norm_main @ f_norm_main.T
        s_cc = c_norm @ c_norm.T

        # Child mask
        valid_child = (parent >= 0)
        child_mask = torch.zeros_like(s_cf, dtype=torch.bool)
        if valid_child.any():
            fine_idx = torch.arange(self.num_fine, device=device)[valid_child]
            child_mask[parent[valid_child], fine_idx] = True
        has_child = child_mask.any(dim=1)

        # ===== Main loss (S3 / S1 / S2) =====
        # S3
        s3_logits = (s_cf / self.tau).masked_fill(~child_mask, float("-inf"))
        lse_s3 = torch.logsumexp(s3_logits, dim=1)
        num_children = child_mask.sum(dim=1).float()
        if self.reg:
            lse_s3 = lse_s3 - torch.log(num_children.clamp(min=1e-8))

        # S1
        eye_c = torch.eye(self.num_coarse, dtype=torch.bool, device=device)
        s1_logits = s_cc / self.tau
        s1_logits = s1_logits.masked_fill(eye_c, float("-inf"))
        lse_s1 = torch.logsumexp(s1_logits, dim=1)

        # S2
        s2_logits_full = s_ff_main / self.tau
        lse_s2 = torch.full((self.num_coarse,), float("-inf"), device=device, dtype=s2_logits_full.dtype)

        for k in range(self.num_coarse):
            child_indices = torch.where((parent == k) & valid_child)[0]
            n_k = int(child_indices.numel())
            if n_k >= 2:
                sub = s2_logits_full[child_indices][:, child_indices]
                eye_f = torch.eye(n_k, dtype=torch.bool, device=device)
                sub = sub.masked_fill(eye_f, float("-inf"))
                lse_k = torch.logsumexp(sub.reshape(-1), dim=0)
                if self.reg:
                    denom = float(n_k * (n_k - 1))
                    lse_k = lse_k - torch.log(torch.tensor(denom, device=device, dtype=sub.dtype))
                lse_s2[k] = lse_k

        logw_s1 = self._log_weighted_sumexp(lse_s1, device=device)
        logw_s2 = self._log_weighted_sumexp(lse_s2, device=device)
        stacked = torch.stack([lse_s3, logw_s1, logw_s2], dim=0)
        log_denom = torch.logsumexp(stacked, dim=0)

        loss_k = log_denom - lse_s3
        if has_child.any():
            loss_main = loss_k[has_child].mean()
        else:
            loss_main = torch.tensor(0.0, device=device, dtype=fine_embeddings.dtype)

        # ===== Sibling margin loss =====
        lambda_mode_scale = self.dist_lambda_scale
        lambda_eff = lambda_mode_scale

        loss_sib = torch.tensor(0.0, device=device, dtype=fine_embeddings.dtype)

        total_pairs_used = 0
        sum_cos = 0.0
        sum_dist = 0.0
        sum_viol = 0.0
        sum_pen = torch.tensor(0.0, device=device, dtype=fine_embeddings.dtype)

        if lambda_eff > 0.0:
            for k in range(self.num_coarse):
                child_indices = torch.where((parent == k) & valid_child)[0]
                n_k = int(child_indices.numel())
                if n_k < 2:
                    continue

                iu = torch.triu_indices(n_k, n_k, offset=1, device=device)

                r = raw_fine_embeddings.to(device)[child_indices]  # [n_k, D]

                diffs = r[iu[0]] - r[iu[1]]              # [P, D]
                d2_pairs = (diffs * diffs).sum(dim=-1)   # [P] = ||ri-rj||^2

                x = (self.margin * self.margin) - d2_pairs
                pen = self._apply_penalty(x)

                r_unit = F.normalize(r, dim=-1)
                cos_pairs = (r_unit[iu[0]] * r_unit[iu[1]]).sum(dim=-1).clamp(-1.0, 1.0)

                # aggregation (loss)
                sum_pen = sum_pen + pen.sum()

                with torch.no_grad():
                    p = int(pen.numel())
                    total_pairs_used += p
                    sum_cos += float(cos_pairs.sum().item())
                    sum_dist += float(torch.sqrt(d2_pairs + 1e-8).sum().item())
                    sum_viol += float((x > 0).float().sum().item())

            # finalize loss_sib
            if total_pairs_used > 0:
                loss_sib = sum_pen / float(total_pairs_used)

        loss = loss_main + lambda_eff * loss_sib

        denom_pairs = max(total_pairs_used, 1)
        diags = {
            "loss": float(loss.item()),
            "loss_main": float(loss_main.item()),
            "loss_sib": float(loss_sib.item()),
            "sib_n_pairs_used": int(total_pairs_used),

            "sib_avg_cos": float(sum_cos / denom_pairs) if total_pairs_used > 0 else 0.0,
            "sib_avg_dist": float(sum_dist / denom_pairs) if total_pairs_used > 0 else 0.0,
            "sib_viol_rate": float(sum_viol / denom_pairs) if total_pairs_used > 0 else 0.0,
        }

        return loss, diags

# --------------------------------------------
# 2) ITAL: Image-Text Alignment Loss
# --------------------------------------------
class AlignmentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self,
                image_features: torch.Tensor,
                text_embeddings: torch.Tensor,
                target_indices: torch.LongTensor) -> torch.Tensor:
        """
        image_features: (B, D)
        text_embeddings: (N, D)
        target_indices: (B,)
        """
        img_feats = F.normalize(image_features, dim=-1)
        text_feats = F.normalize(text_embeddings, dim=-1)

        logits = (img_feats @ text_feats.t()) / max(1e-12, self.temperature)
        return F.cross_entropy(logits, target_indices)


# --------------------------------------------
# 3) CL: Supervised Contrastive Loss (for images, label=string)
# --------------------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = float(temperature)

    @staticmethod
    def _labels_to_ids(labels: List[str], device):
        if not (isinstance(labels, (list, tuple)) and len(labels) > 0 and isinstance(labels[0], str)):
            raise TypeError("Labels should be a non-empty list of strings.")
        label_to_id: Dict[str, int] = {}
        ids = []
        for l in labels:
            if l not in label_to_id:
                label_to_id[l] = len(label_to_id)
            ids.append(label_to_id[l])
        return torch.tensor(ids, device=device, dtype=torch.long)

    def forward(self, features: torch.Tensor, labels: List[str]) -> torch.Tensor:
        """
        features: (B, D)
        labels:   list[str] (length B)
        """
        device = features.device
        B = features.size(0)

        z = F.normalize(features, dim=1)
        sim = (z @ z.T) / max(1e-6, self.temperature)  # (B, B)

        eye = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(eye, torch.finfo(sim.dtype).min) 

        ids = self._labels_to_ids(labels, device)
        pos_mask = ids.unsqueeze(1).eq(ids.unsqueeze(0)) & (~eye)

        if not pos_mask.any():
            return torch.tensor(0.0, device=device, dtype=features.dtype)

        log_prob = F.log_softmax(sim, dim=1)  # (B, B)

        pos_weight = pos_mask.float()
        pos_count = pos_weight.sum(dim=1)

        loss_samples = -(log_prob * pos_weight).sum(dim=1)
        loss_samples = torch.where(pos_count > 0,
                                   loss_samples / pos_count.clamp(min=1),
                                   torch.zeros_like(loss_samples))

        return loss_samples.mean()
