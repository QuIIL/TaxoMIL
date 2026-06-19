import os
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score

from utils.utils import seed_torch, patch_unembedding_to_tokenizer, get_model_path
from decoder import build_model
from .train import get_dataset_pack

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@torch.no_grad()
def test(
    model,
    dataloader,
    device,
    tokenizer,
):
    model.eval()

    all_coarse_preds = []
    all_coarse_targets = []
    all_fine_preds = []
    all_fine_targets = []

    for batch in tqdm(dataloader, desc="Testing"):
        if batch[0] is None:
            continue

        img_emb, c_tgt, f_tgt = batch
        img_emb = img_emb.to(device)
        c_tgt = c_tgt.to(device)
        f_tgt = f_tgt.to(device)

        # Get predictions
        c_texts_pred, f_texts_pred, _, _ = model(img_emb)

        # Convert targets to strings
        c_tgt_strs = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in c_tgt]
        f_tgt_strs = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in f_tgt]

        all_coarse_preds.extend([str(p).strip() for p in c_texts_pred])
        all_coarse_targets.extend([str(t).strip() for t in c_tgt_strs])
        all_fine_preds.extend([str(p).strip() for p in f_texts_pred])
        all_fine_targets.extend([str(t).strip() for t in f_tgt_strs])

    return all_coarse_preds, all_coarse_targets, all_fine_preds, all_fine_targets


def calculate_metrics(preds, targets):
    """Calculate accuracy, balanced accuracy, weighted F1, and kappa score."""
    if not targets:
        return 0.0, 0.0, 0.0

    correct = [1 if p == t else 0 for p, t in zip(preds, targets)]
    accuracy = sum(correct) / len(correct) * 100

    unique_labels = sorted(set(targets) | set(preds))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    target_ids = [label_to_idx[t] for t in targets]
    pred_ids = [label_to_idx[p] for p in preds]

    weighted_f1 = f1_score(target_ids, pred_ids, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(target_ids, pred_ids)

    return accuracy, weighted_f1, kappa


def run_test(args, pack, device):
    """Run test evaluation."""
    # Build model
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
    ).to(device)

    tokenizer = model.tokenizer
    patch_unembedding_to_tokenizer(model, tokenizer)

    # Load checkpoint
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    print(f"Loading checkpoint from: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully")

    # Load test dataset
    _, _, test_ds = pack.get_datasets_fn(args, tokenizer)

    if test_ds is None:
        print("Warning: Test dataset is None, trying to use val dataset instead")
        _, test_ds, _ = pack.get_datasets_fn(args, tokenizer)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pack.custom_collate_fn,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # Run test
    print("\nRunning evaluation on test set...")
    c_preds, c_targets, f_preds, f_targets = test(model, test_loader, device, tokenizer)

    # Print detailed outputs
    if args.print_output:
        print("\n" + "="*80)
        print("DETAILED PREDICTIONS AND TARGETS")
        print("="*80)
        for i, (c_pred, c_target, f_pred, f_target) in enumerate(zip(c_preds, c_targets, f_preds, f_targets)):
            c_match = "✓" if c_pred == c_target else "✗"
            f_match = "✓" if f_pred == f_target else "✗"
            print(f"{i+1:4d}. {c_match} Coarse: {c_pred:50s} | {c_target}")
            print(f"      {f_match} Fine:   {f_pred:50s} | {f_target}")

    # Calculate metrics for coarse labels
    print("\n" + "="*60)
    print("COARSE")
    print("="*60)
    c_acc, c_wf1, c_kappa = calculate_metrics(c_preds, c_targets)
    print(f"Accuracy:         {c_acc:.2f}%")
    print(f"Weighted F1 Score: {c_wf1:.4f}")
    print(f"Kappa Score:      {c_kappa:.4f}")

    # Calculate metrics for fine labels
    print("\n" + "="*60)
    print("FINE")
    print("="*60)
    f_acc, f_wf1, f_kappa = calculate_metrics(f_preds, f_targets)
    print(f"Accuracy:         {f_acc:.2f}%")
    print(f"Weighted F1 Score: {f_wf1:.4f}")
    print(f"Kappa Score:      {f_kappa:.4f}")

    # holistic score (both coarse and fine must be correct)
    print("\n" + "="*60)
    print("HOLISTIC (COARSE + FINE)")
    print("="*60)
    holistic_preds = [c + "|" + f for c, f in zip(c_preds, f_preds)]
    holistic_targets = [c + "|" + f for c, f in zip(c_targets, f_targets)]
    holistic_acc, holistic_wf1, holistic_kappa = calculate_metrics(holistic_preds, holistic_targets)

    print(f"Accuracy:         {holistic_acc:.2f}%")
    print(f"Weighted F1 Score: {holistic_wf1:.4f}")
    print(f"Kappa Score:      {holistic_kappa:.4f}")
