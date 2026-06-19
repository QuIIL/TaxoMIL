import os
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from utils.loss_scheduler import PeriodicLossScheduler
from utils.loss import (
    TextHierarchicalLoss,
    AlignmentLoss,
    ContrastiveLoss,
)
from utils.utils import (
    seed_torch,
    get_optimizer,
    get_model_path,
    patch_unembedding_to_tokenizer,
    build_text_bank_for_losses,
    calculate_accuracies_shared_dual
)
from decoder import build_model
from dataloader.dataset_labels import DatasetLabels

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------
# Dataset/labels config
# -------------------------
@dataclass
class DatasetPack:
    all_coarse_labels: List[str]
    all_fine_labels: List[str]
    hierarchy_map: Dict[str, str]
    custom_collate_fn: any
    get_datasets_fn: any


def get_dataset_pack(data_name: str) -> DatasetPack:
    data_name_upper = str(data_name).upper()

    dataset_modules = {
        "BRACS": "dataloader.dataset",
        "PANDA": "dataloader.dataset",
    }

    if data_name_upper not in dataset_modules:
        raise NotImplementedError(f"Dataset {data_name} not implemented.")

    module_path = dataset_modules[data_name_upper]
    module = __import__(module_path, fromlist=["get_datasets", "custom_collate_fn"])
    get_datasets = getattr(module, "get_datasets")
    custom_collate_fn = getattr(module, "custom_collate_fn")

    labels_config = DatasetLabels.get_dataset_config(data_name)

    return DatasetPack(
        all_coarse_labels=labels_config["coarse"],
        all_fine_labels=labels_config["fine"],
        hierarchy_map=labels_config["hierarchy"],
        custom_collate_fn=custom_collate_fn,
        get_datasets_fn=get_datasets,
    )

# -------------------------
# Loss computation helpers
# -------------------------
def compute_ital_loss(criteria, head_type: str, features: torch.Tensor, text_bank: Dict[str, torch.Tensor],
                      ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    valid = (ids != -1)
    if valid.any():
        return criteria[f"{head_type}_ital"](features[valid], text_bank[f"proj_{head_type}"], ids[valid])
    return torch.tensor(0.0, device=device)

def compute_optional_losses(
    criteria: Dict[str, nn.Module],
    use_flags: Dict[str, bool],
    coarse_vis: torch.Tensor,
    fine_vis: torch.Tensor,
    text_bank: Dict[str, torch.Tensor],
    c_strs_gt: List[str],
    f_strs_gt: List[str],
    coarse_ids: torch.Tensor,
    fine_ids: torch.Tensor,
    device: torch.device,
) -> tuple:
    L_HIER = torch.tensor(0.0, device=device)
    L_ITAL = torch.tensor(0.0, device=device)
    L_CL = torch.tensor(0.0, device=device)
    hier_diags = {}

    if use_flags.get("HIER", False):
        L_HIER, hier_diags = criteria["hier"](text_bank["proj_fine"], text_bank["proj_coarse"], text_bank["raw_fine"])

    if use_flags.get("ITAL", False):
        ital_c = compute_ital_loss(criteria, "coarse", coarse_vis, text_bank, coarse_ids, device)
        ital_f = compute_ital_loss(criteria, "fine", fine_vis, text_bank, fine_ids, device)
        L_ITAL = ital_c + ital_f

    if use_flags.get("CL", False):
        L_CL = criteria["c_cl"](coarse_vis, c_strs_gt) + criteria["f_cl"](fine_vis, f_strs_gt)

    return L_HIER, L_ITAL, L_CL, hier_diags

# -------------------------
# Train / Validate
# -------------------------
def train_one_epoch(
    epoch: int,
    model,
    criteria: Dict[str, nn.Module],
    optimizer,
    dataloader,
    device,
    tokenizer,
    args,
    label_maps: Dict[str, Dict[str, int]],
    dyn_weights: Dict[str, float],
    all_coarse_labels: List[str],
    all_fine_labels: List[str],
):
    model.train()
    loss_sums = {"ce": 0.0, "hier": 0.0, "ital": 0.0, "cl": 0.0}
    hier_diags_sums = {}

    use_flags = {"HIER": args.use_HIER, "ITAL": args.use_ITAL, "CL": args.use_CL}

    for batch in tqdm(dataloader, desc=f"Train Epoch {epoch} "):
        if batch[0] is None:
            continue

        img_emb, c_tgt, f_tgt = batch
        img_emb = img_emb.to(device)
        c_tgt = c_tgt.to(device)
        f_tgt = f_tgt.to(device)

        optimizer.zero_grad(set_to_none=True)

        # --- CE (generation) ---
        tf_out = model.forward_train(img_emb, c_tgt, f_tgt)
        c_logits = tf_out["coarse_logits"]
        f_logits = tf_out["fine_logits"]

        L_CE_c = criteria["gen"](c_logits.flatten(0, 1), c_tgt.flatten())
        L_CE_f = criteria["gen"](f_logits.flatten(0, 1), f_tgt.flatten())
        ce_loss = (L_CE_c + L_CE_f)

        loss_sums["ce"] += float(ce_loss.item())

        # --- optional losses ---
        L_HIER = torch.tensor(0.0, device=device)
        L_ITAL = torch.tensor(0.0, device=device)
        L_CL = torch.tensor(0.0, device=device)
        hier_diags = {}

        if any(use_flags.values()):
            text_bank_local = None
            if use_flags["HIER"] or use_flags["ITAL"]:
                text_bank_local = build_text_bank_for_losses(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    all_coarse_labels=all_coarse_labels,
                    all_fine_labels=all_fine_labels,
                )
            impl = getattr(model, "impl", model)
            coarse_vis, fine_vis = impl.encode_image_features(img_emb)

            c_strs_gt = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in c_tgt]
            f_strs_gt = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in f_tgt]
            coarse_ids = torch.tensor([label_maps["coarse"].get(s, -1) for s in c_strs_gt], device=device, dtype=torch.long)
            fine_ids = torch.tensor([label_maps["fine"].get(s, -1) for s in f_strs_gt], device=device, dtype=torch.long)

            L_HIER, L_ITAL, L_CL, hier_diags = compute_optional_losses(
                criteria, use_flags, coarse_vis, fine_vis, text_bank_local,
                c_strs_gt, f_strs_gt, coarse_ids, fine_ids, device
            )
            loss_sums["hier"] += float(L_HIER.item())
            loss_sums["ital"] += float(L_ITAL.item()) if torch.is_tensor(L_ITAL) else float(L_ITAL)
            loss_sums["cl"] += float(L_CL.item()) if torch.is_tensor(L_CL) else float(L_CL)

            for k, v in hier_diags.items():
                v_val = float(v.item()) if torch.is_tensor(v) else float(v)
                hier_diags_sums.setdefault(k, 0.0)
                hier_diags_sums[k] += v_val

        batch_loss = (
            dyn_weights["ce"] * ce_loss
            + dyn_weights["hier"] * L_HIER
            + dyn_weights["ital"] * L_ITAL
            + dyn_weights["cl"] * L_CL
        )

        batch_loss.backward()
        optimizer.step()

    num_batches = max(1, len(dataloader))
    avg_losses = {k: v / num_batches for k, v in loss_sums.items()}

    if hier_diags_sums:
        avg_hier_diags = {k: v / num_batches for k, v in hier_diags_sums.items()}
    else:
        avg_hier_diags = {}

    print(
        f"[Epoch {epoch} Train] "
        f"CE={avg_losses['ce']:.4f} HIER={avg_losses['hier']:.4f} ITAL={avg_losses['ital']:.4f} CL={avg_losses['cl']:.4f}"
    )
    if avg_hier_diags:
        print("  HIER Diags:")
        for k, v in avg_hier_diags.items():
            print(f"    {k}={v:.4f}")

    if args.wandb:
        log_data = {f"train_{k}_loss": v for k, v in avg_losses.items()}
        log_data.update({f"train_hier_{k}": v for k, v in avg_hier_diags.items()})
        wandb.log(log_data, step=epoch)


@torch.no_grad()
def validate(
    epoch: int,
    model,
    criteria: Dict[str, nn.Module],
    dataloader,
    device,
    tokenizer,
    args,
    label_maps: Dict[str, Dict[str, int]],
    dyn_weights: Dict[str, float],
    text_bank: Dict[str, torch.Tensor],
):
    model.eval()
    loss_sums = {"ce": 0.0, "hier": 0.0, "ital": 0.0, "cl": 0.0}
    hier_diags_sums = {}

    acc_c = 0
    acc_f = 0
    total = 0

    use_flags = {"HIER": args.use_HIER, "ITAL": args.use_ITAL, "CL": args.use_CL}

    for batch in tqdm(dataloader, desc=f"Validate Epoch {epoch} "):
        if batch[0] is None:
            continue

        img_emb, c_tgt, f_tgt = batch
        img_emb = img_emb.to(device)
        c_tgt = c_tgt.to(device)
        f_tgt = f_tgt.to(device)

        # --- Accuracy ---
        c_texts_pred, f_texts_pred, _, _ = model(img_emb)
        cc, fc, n = calculate_accuracies_shared_dual(c_texts_pred, f_texts_pred, c_tgt, f_tgt, tokenizer)
        acc_c += cc
        acc_f += fc
        total += n

        # --- CE loss ---
        tf_out = model.forward_train(img_emb, c_tgt, f_tgt)
        c_logits = tf_out["coarse_logits"]
        f_logits = tf_out["fine_logits"]
        L_CE_c = criteria["gen"](c_logits.flatten(0, 1), c_tgt.flatten())
        L_CE_f = criteria["gen"](f_logits.flatten(0, 1), f_tgt.flatten())
        ce_loss = (L_CE_c + L_CE_f)

        loss_sums["ce"] += float(ce_loss.item())

        # --- optional losses ---
        L_HIER = torch.tensor(0.0, device=device)
        L_ITAL = torch.tensor(0.0, device=device)
        L_CL = torch.tensor(0.0, device=device)
        hier_diags = {}

        if any(use_flags.values()):
            impl = getattr(model, "impl", model)
            coarse_vis, fine_vis = impl.encode_image_features(img_emb)

            c_strs_gt = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in c_tgt]
            f_strs_gt = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in f_tgt]
            coarse_ids = torch.tensor([label_maps["coarse"].get(s, -1) for s in c_strs_gt], device=device, dtype=torch.long)
            fine_ids = torch.tensor([label_maps["fine"].get(s, -1) for s in f_strs_gt], device=device, dtype=torch.long)

            L_HIER, L_ITAL, L_CL, hier_diags = compute_optional_losses(
                criteria, use_flags, coarse_vis, fine_vis, text_bank,
                c_strs_gt, f_strs_gt, coarse_ids, fine_ids, device
            )
            loss_sums["hier"] += float(L_HIER.item())
            loss_sums["ital"] += float(L_ITAL.item()) if torch.is_tensor(L_ITAL) else float(L_ITAL)
            loss_sums["cl"] += float(L_CL.item()) if torch.is_tensor(L_CL) else float(L_CL)

            for k, v in hier_diags.items():
                v_val = float(v.item()) if torch.is_tensor(v) else float(v)
                hier_diags_sums.setdefault(k, 0.0)
                hier_diags_sums[k] += v_val

    num_batches = max(1, len(dataloader))
    avg_losses = {k: v / num_batches for k, v in loss_sums.items()}

    if hier_diags_sums:
        avg_hier_diags = {k: v / num_batches for k, v in hier_diags_sums.items()}
    else:
        avg_hier_diags = {}

    if total > 0:
        acc_coarse = acc_c / total * 100.0
        acc_fine = acc_f / total * 100.0
        score = (acc_coarse + acc_fine) / 2.0
    else:
        acc_coarse = 0.0
        acc_fine = 0.0
        score = 0.0

    val_combined = (
        dyn_weights["ce"] * avg_losses["ce"]
        + dyn_weights["hier"] * avg_losses["hier"]
        + dyn_weights["ital"] * avg_losses["ital"]
        + dyn_weights["cl"] * avg_losses["cl"]
    )

    print(
        f"[Epoch {epoch} Val] "
        f"AccC={acc_coarse:.2f}% AccF={acc_fine:.2f}% | "
        f"CE={avg_losses['ce']:.4f} HIER={avg_losses['hier']:.4f} ITAL={avg_losses['ital']:.4f} CL={avg_losses['cl']:.4f} | "
        f"Combined(w)={val_combined:.4f} | Score={score:.2f}"
    )
    if avg_hier_diags:
        print("  HIER Diags:")
        for k, v in avg_hier_diags.items():
            print(f"    {k}={v:.4f}")

    if args.wandb:
        log_dict = {f"val_{k}_loss": v for k, v in avg_losses.items()}
        log_dict.update(
            {
                "val_coarse_acc": acc_coarse,
                "val_fine_acc": acc_fine,
                "val_combined_weighted": val_combined,
                "val_score": score,
            }
        )
        log_dict.update({f"val_hier_{k}": v for k, v in avg_hier_diags.items()})
        wandb.log(log_dict, step=epoch)

    return score, avg_losses["ce"], acc_coarse, acc_fine


# -------------------------
# Experiment Runner
# -------------------------
def run_experiment(
    args,
    pack: DatasetPack,
    device: torch.device,
):
    lr = args.lr

    # load prompt templates
    with open(args.prompt_templates_file, 'r') as f:
        prompt_config = json.load(f)
    prompt_templates = prompt_config.get("base_templates", [])

    # build model
    model_path = get_model_path(args.decoder_type)
    model = build_model(
        model_path=model_path,
        device=str(device),
        dataset=args.data,
        emb_dim=args.emb_dim,
        gated=True,
        output_len_coarse=args.output_len_coarse,
        output_len_fine=args.output_len_fine,
        all_coarse_labels=pack.all_coarse_labels,
        all_fine_labels=pack.all_fine_labels,
        prompt_templates=prompt_templates,
    ).to(device)

    tokenizer = model.tokenizer

    patch_unembedding_to_tokenizer(model, tokenizer)

    # label maps
    label_maps = {
        "coarse": {lbl: i for i, lbl in enumerate(pack.all_coarse_labels)},
        "fine": {lbl: i for i, lbl in enumerate(pack.all_fine_labels)},
    }

    # data
    train_ds, val_ds, _ = pack.get_datasets_fn(args, tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pack.custom_collate_fn,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pack.custom_collate_fn,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # criteria
    criteria = {
        "gen": nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id),
    }

    use_HIER = args.use_HIER
    use_ITAL = args.use_ITAL
    use_CL = args.use_CL

    # HIER
    if use_HIER:
        print('Using Margin-based Text Hierarchical Loss (metric: dist2).')
        criteria["hier"] = TextHierarchicalLoss(
            pack.all_fine_labels,
            pack.all_coarse_labels,
            pack.hierarchy_map,
            tau=args.hier_temp,
            margin=args.margin,
        ).to(device)

    # ITAL
    if use_ITAL:
        criteria["c_ital"] = AlignmentLoss(temperature=args.c_temp).to(device)
        criteria["f_ital"] = AlignmentLoss(temperature=args.f_temp).to(device)

    # CL
    if use_CL:
        criteria["c_cl"] = ContrastiveLoss(temperature=args.c_temp).to(device)
        criteria["f_cl"] = ContrastiveLoss(temperature=args.f_temp).to(device)

    # optimizer & scheduler
    optimizer = get_optimizer(args.optimizer, model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # run name / ckpt dir
    run_name = f"{args.data}_{args.decoder_type}_b{args.batch_size}_s{args.seed}_lr{lr}"

    if args.use_base:
        run_name += "_BASE"
    else:
        if use_HIER:
            run_name += f"_HIER_MARGIN_m{args.margin}_w{args.hier_weight}_t{args.hier_temp}"
        if use_ITAL:
            run_name += f"_ITAL_w{args.ital_weight}_ct{args.c_temp}_ft{args.f_temp}"
        if use_CL:
            run_name += f"_CL_w{args.cl_weight}_t{args.c_temp}"

    if args.dyn_weight:
        run_name += f"_periodic{args.n_cycles}"

    ckpt_dir = os.path.join(args.save_dir, args.data)
    os.makedirs(ckpt_dir, exist_ok=True)

    # wandb init
    if args.wandb:
        wandb.init(project=args.wandb_project, name=run_name, reinit=True)
        wandb.config.update(vars(args))

    # dyn scheduler
    lw_sched = None
    if args.dyn_weight:
        lw_sched = PeriodicLossScheduler(
            hier_max=args.hier_weight,
            ital_max=args.ital_weight,
            cl_max=args.cl_weight,
            n_cycles=args.n_cycles,
        )

    best_score = -1e9
    patience_cnt = 0
    prev_ckpt = None

    for epoch in range(1, args.epochs + 1):
        if args.dyn_weight and lw_sched is not None:
            w_ce, w_hier, w_ital, w_cl = lw_sched(epoch - 1)
        else:
            w_ce = 1.0
            w_hier = args.hier_weight if use_HIER else 0.0
            w_ital = args.ital_weight if use_ITAL else 0.0
            w_cl = args.cl_weight if use_CL else 0.0

        dyn_weights = {"ce": w_ce, "hier": w_hier, "ital": w_ital, "cl": w_cl}

        if args.wandb:
            wandb.log({"w_ce": w_ce, "w_hier": w_hier, "w_ital": w_ital, "w_cl": w_cl}, step=epoch)

        if use_HIER or use_ITAL:
            text_bank = build_text_bank_for_losses(
                model=model,
                tokenizer=tokenizer,
                device=device,
                all_coarse_labels=pack.all_coarse_labels,
                all_fine_labels=pack.all_fine_labels,
            )
        else:
            text_bank = None

        train_one_epoch(
            epoch=epoch,
            model=model,
            criteria=criteria,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            tokenizer=tokenizer,
            args=args,
            label_maps=label_maps,
            dyn_weights=dyn_weights,
            all_coarse_labels=pack.all_coarse_labels,
            all_fine_labels=pack.all_fine_labels,
        )
        score, _, val_c_acc, val_f_acc = validate(
            epoch=epoch,
            model=model,
            criteria=criteria,
            dataloader=val_loader,
            device=device,
            tokenizer=tokenizer,
            args=args,
            label_maps=label_maps,
            dyn_weights=dyn_weights,
            text_bank=text_bank,
        )

        scheduler.step()

        if score > best_score:
            best_score = score
            patience_cnt = 0
            if prev_ckpt and os.path.exists(prev_ckpt):
                os.remove(prev_ckpt)
            save_path = os.path.join(ckpt_dir, f"{run_name}.pth")
            torch.save(model.state_dict(), save_path)
            prev_ckpt = save_path
            print(f">>> New best saved: {save_path} | best_score={best_score:.2f}")
        elif not args.no_early_stopping:
            patience_cnt += 1
            print(f"Patience: {patience_cnt}/{args.patience}")
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    if args.wandb:
        wandb.finish()

    print(f"[DONE] {run_name} | best_score={best_score:.2f}")
    return best_score

