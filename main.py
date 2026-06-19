import argparse
import torch
import warnings
import os

from utils.utils import seed_torch
from trainer import get_dataset_pack, run_experiment, run_test

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    p = argparse.ArgumentParser(description="Main entry point for training and testing")

    # Common arguments
    p.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                   help="Mode: train or test")
    p.add_argument("--gpu", type=int, default=0, help="GPU device index")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--data", type=str, default="BRACS",
                   choices=["BRACS", "PANDA"],
                   help="Dataset name")
    p.add_argument("--decoder_type", type=str, default="GPT2",
                   choices=["GPT2", "GPT2-medium"],
                   help="Decoder model type")

    p.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    p.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--emb_dim", type=int, default=1024, help="Embedding dimension")
    p.add_argument("--csv_path", type=str, default=None, help="Optional dataset CSV path")
    p.add_argument("--verify_files", action="store_true", help="Verify feature files before training/testing")
    
    p.add_argument("--output_len_coarse", type=int, default=5, help="Output length for coarse labels")
    p.add_argument("--output_len_fine", type=int, default=15, help="Output length for fine labels")

    # Training arguments
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adam", "adamw", "sgd"], help="Optimizer type")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping")

    p.add_argument("--prompt_templates_file", type=str, default="config/prompt_templates.json",
                   help="Path to JSON file containing prompt templates")

    # Loss flags
    p.add_argument("--use_base", action="store_true", default=False,
                   help="Only CE loss (no HIER/ITAL/CL)")
    p.add_argument("--use_HIER", action="store_true", default=False, help="Use hierarchical loss")
    p.add_argument("--use_ITAL", action="store_true", default=False, help="Use alignment loss")
    p.add_argument("--use_CL", action="store_true", default=False, help="Use contrastive loss")
    p.add_argument("--use_all_loss", action="store_true", default=False,
                   help="Use all losses (HIER + ITAL + CL)")

    # Margin and temperature parameters
    p.add_argument("--margin", type=float, default=1.5, help="Margin for hierarchical loss")
    p.add_argument("--hier_temp", type=float, default=1.0, help="Hierarchical loss temperature")
    p.add_argument("--hier_weight", type=float, default=0.3, help="Hierarchical loss weight")
    p.add_argument("--ital_weight", type=float, default=0.3, help="Alignment loss weight")
    p.add_argument("--cl_weight", type=float, default=0.1, help="Contrastive loss weight")
    p.add_argument("--c_temp", type=float, default=0.07, help="Coarse loss temperature")
    p.add_argument("--f_temp", type=float, default=0.07, help="Fine loss temperature")


    # Dynamic weight scheduling
    p.add_argument("--dyn_weight", action="store_true", default=False,
                   help="Use periodic weight scheduling")
    p.add_argument("--n_cycles", type=int, default=3, help="Number of weight cycles")

    # I/O and logging
    p.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint save directory")
    p.add_argument("--wandb_project", type=str, default="TaxoMIL", help="W&B project name")
    p.add_argument("--wandb", action="store_true", default=False, help="Enable W&B logging")

    # Testing arguments
    p.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for testing")
    p.add_argument("--print_output", action="store_true", help="Print detailed predictions in test mode")

    args = p.parse_args()
    _process_loss_flags(args)
    return args

def _process_loss_flags(args):
    if args.use_base:
        args.use_HIER = False
        args.use_ITAL = False
        args.use_CL = False

    if args.use_all_loss:
        args.use_HIER = True
        args.use_ITAL = True
        args.use_CL = True
        args.dyn_weight = True
        print('INFO: --use_all_loss enabled -> auto enabling HIER(margin) + ITAL + CL + dyn_weight')

    if args.use_ITAL and not args.use_HIER:
        print('INFO: --use_ITAL enabled -> auto enabling --use_HIER.')
        args.use_HIER = True


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    seed_torch(args.seed, device)

    pack = get_dataset_pack(args.data)

    if args.mode == "train":
        print("=" * 80)
        best = run_experiment(args=args, pack=pack, device=device)

        print("\n=== SUMMARY ===")
        print(f"shared: best_score={best:.2f}")

    elif args.mode == "test":
        if not args.ckpt_path:
            raise ValueError("--ckpt_path is required for test mode")

        print("=" * 80)
        run_test(args=args, pack=pack, device=device)


if __name__ == "__main__":
    main()
