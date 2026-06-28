import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CSV_PATHS = {
    "BRACS": "dataset_csv/BRACS_with_path.csv",
    "PANDA": "dataset_csv/PANDA_with_path.csv",
}

def _infer_patch_count(loaded):
    try:
        if isinstance(loaded, torch.Tensor):
            return int(loaded.size(0)) if loaded.dim() >= 1 else 1
        if isinstance(loaded, dict):
            for v in loaded.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 2:
                    return int(v.size(0))
            for v in loaded.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    return int(v.size(0))
            for v in loaded.values():
                if isinstance(v, (list, tuple)):
                    return len(v)
            return 1
        if isinstance(loaded, (list, tuple)):
            return len(loaded)
    except Exception:
        pass
    return 1


def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch:
        return None, None, None

    image_embeddings, coarse_ids_list, fine_ids_list = zip(*batch)

    image_embeddings = pad_sequence(image_embeddings, batch_first=True, padding_value=0)
    coarse_ids = pad_sequence(coarse_ids_list, batch_first=True, padding_value=-100)
    fine_ids = pad_sequence(fine_ids_list, batch_first=True, padding_value=-100)
    return image_embeddings, coarse_ids, fine_ids


def _resolve_csv_path(dataset_name: str, csv_path: str | None):
    if csv_path:
        return csv_path
    ds = str(dataset_name).upper()
    if ds not in DEFAULT_CSV_PATHS:
        raise ValueError(f"Unknown dataset '{ds}'. Allowed: {list(DEFAULT_CSV_PATHS.keys())}")
    return DEFAULT_CSV_PATHS[ds]


class UNI_PathDataset(Dataset):
    """
    Unified CSV schema:
      - split: train/val/test
      - coarse_label
      - fine_label
      - feature_path: .pt file path
    """

    def __init__(
        self,
        dataset_name: str,
        csv_path: str,
        split: str,
        tokenizer,
        path_col: str = "feature_path",
        split_col: str = "split",
        coarse_col: str = "coarse_label",
        fine_col: str = "fine_label",
        verify_files: bool = True,
        output_len_coarse: int = 5,
        output_len_fine: int = 15,
        lowercase_coarse: bool = True,
        lowercase_fine: bool = True,
    ):
        if not csv_path:
            raise ValueError("csv_path is empty. Provide args.csv_path or use dataset default path.")

        df = pd.read_csv(csv_path)
        
        # column check
        for c in [split_col, coarse_col, fine_col, path_col]:
            if c not in df.columns:
                raise KeyError(f"CSV missing column '{c}'. Columns: {list(df.columns)}")

        # standardize split
        df[split_col] = df[split_col].astype(str).str.strip().str.lower()
        self.data = df[df[split_col] == split].reset_index(drop=True)

        self.dataset_name = str(dataset_name).upper()
        self.csv_path = csv_path

        self.tokenizer = tokenizer
        self.path_col = path_col
        self.split_col = split_col
        self.coarse_col = coarse_col
        self.fine_col = fine_col

        self.output_len_coarse = int(output_len_coarse)
        self.output_len_fine = int(output_len_fine)
        self.lowercase_coarse = bool(lowercase_coarse)
        self.lowercase_fine = bool(lowercase_fine)

        self.invalid_files = set()
        self.valid_indices = []
        self.patch_counts = []

        if verify_files:
            self._verify_all_files(split)
        else:
            self.valid_indices = list(range(len(self.data)))

    def _get_pt_path(self, row):
        p = row.get(self.path_col, None)
        if pd.isna(p) or p is None:
            return None
        p = str(p).strip()
        return p if p else None

    def _prepare_label(self, row):
        coarse = str(row[self.coarse_col]).strip()
        fine = str(row[self.fine_col]).strip()

        if self.lowercase_coarse:
            coarse = coarse.lower()

        if self.lowercase_fine and self.dataset_name != "PANDA":
            fine = fine.lower()

        return coarse, fine

    def _verify_all_files(self, split_name: str):
        print(f"\nVerifying pt files for split='{split_name}' ...")
        for idx in tqdm(range(len(self.data)), desc="Verifying data files"):
            row = self.data.iloc[idx]
            pt_path = self._get_pt_path(row)

            if pt_path is None or (not os.path.exists(pt_path)):
                self.invalid_files.add(pt_path)
                continue

            try:
                loaded = torch.load(pt_path, map_location="cpu")
                self.valid_indices.append(idx)
                self.patch_counts.append(_infer_patch_count(loaded))
            except Exception:
                self.invalid_files.add(pt_path)

        print(
            f"Total rows checked: {len(self.data)}, "
            f"Valid: {len(self.valid_indices)}, Invalid: {len(self.invalid_files)}"
        )

        if self.patch_counts:
            arr = np.array(self.patch_counts, dtype=np.int32)
            print(
                f"Avg patches per slide: {arr.mean():.2f} "
                f"(min={arr.min()}, median={np.median(arr)}, max={arr.max()}, n={arr.size})"
            )

    def patch_count_stats(self):
        if not self.patch_counts:
            return None
        arr = np.array(self.patch_counts, dtype=np.int32)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "min": int(arr.min()),
            "median": float(np.median(arr)),
            "max": int(arr.max()),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.data.iloc[real_idx]

        pt_path = self._get_pt_path(row)
        if pt_path is None or (not os.path.exists(pt_path)):
            return None, None, None

        try:
            image_embedding = torch.load(pt_path, map_location="cpu")
        except Exception:
            return None, None, None

        coarse_text, fine_text = self._prepare_label(row)

        coarse_ids = self._encode_label(coarse_text, self.output_len_coarse)
        fine_ids = self._encode_label(fine_text, self.output_len_fine)

        return image_embedding, coarse_ids, fine_ids

    def _encode_label(self, text: str, output_len: int):
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max(1, output_len - 1),
            add_special_tokens=False,
        )

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        ids = torch.cat([enc["input_ids"].squeeze(0), torch.tensor([eos_id])], dim=0)
        ids = ids[:output_len]

        if ids.size(0) < output_len:
            pad = torch.full((output_len - ids.size(0),), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)
        return ids


def get_datasets(args, tokenizer):
    dataset_name = getattr(args, "data", "UNKNOWN")
    csv_path = _resolve_csv_path(dataset_name, getattr(args, "csv_path", None))
    verify = bool(getattr(args, "verify_files", False))

    train_ds = UNI_PathDataset(dataset_name, csv_path, "train", tokenizer, verify_files=verify)
    val_ds   = UNI_PathDataset(dataset_name, csv_path, "val",   tokenizer, verify_files=verify)
    test_ds  = UNI_PathDataset(dataset_name, csv_path, "test",  tokenizer, verify_files=verify)
    return train_ds, val_ds, test_ds


def _print_split_distribution(df: pd.DataFrame, split_col: str = "split"):
    df = df.copy()
    df[split_col] = df[split_col].astype(str).str.strip().str.lower()
    vc = df[split_col].value_counts()
    order = ["train", "val", "test"]
    print("\n[Split distribution]")
    for k in order:
        if k in vc:
            print(f"- {k}: {int(vc[k])}")
    for k in vc.index:
        if k not in order:
            print(f"- {k}: {int(vc[k])}")


def _print_patch_stats(title: str, stats: dict | None):
    if not stats:
        print(f"\n[{title}] No patch stats available.")
        return
    print(f"\n[{title}] patches per slide")
    print(
        f"- count: {stats['count']}, "
        f"min/mean/median/max: {stats['min']} / {stats['mean']:.2f} / {stats['median']:.1f} / {stats['max']}"
    )
    print(f"- p90/p95/p99: {stats['p90']:.1f} / {stats['p95']:.1f} / {stats['p99']:.1f}")


def _print_label_distribution_all(
    csv_path: str,
    dataset_name: str,
    split_col: str = "split",
    coarse_col: str = "coarse_label",
    fine_col: str = "fine_label",
):
    df = pd.read_csv(csv_path)

    df[split_col] = df[split_col].astype(str).str.strip().str.lower()
    df[coarse_col] = df[coarse_col].astype(str).str.strip().str.lower()
    df[fine_col] = df[fine_col].astype(str).str.strip()
    if str(dataset_name).upper() != "PANDA":
        df[fine_col] = df[fine_col].str.lower()

    splits = ["train", "val", "test"]

    print("\n" + "=" * 80)
    print("[Label distribution: ALL labels]")

    print("\n[Coarse label counts per split]")
    coarse_tab = (
        df.groupby([split_col, coarse_col]).size().unstack(fill_value=0)
        .reindex(index=splits, fill_value=0)
    )
    print(coarse_tab)

    print("\n[Fine label counts per split]")
    fine_tab = (
        df.groupby([split_col, fine_col]).size().unstack(fill_value=0)
        .reindex(index=splits, fill_value=0)
    )
    print(fine_tab)

    print("\n[Coarse->Fine pair counts (overall)]")
    pair_tab = df.groupby([coarse_col, fine_col]).size().unstack(fill_value=0)
    print(pair_tab)
    print("=" * 80)


def main():
    import argparse
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help=f"dataset name: {list(DEFAULT_CSV_PATHS.keys())}")
    ap.add_argument("--csv_path", type=str, default="", help="override csv path (optional)")
    ap.add_argument("--verify_files", action="store_true", default=True, help="Verify .pt files and print patch statistics")
    args = ap.parse_args()

    dataset_name = str(args.data).upper()
    csv_path = _resolve_csv_path(dataset_name, args.csv_path)

    print(f"\n[INFO] dataset={dataset_name}  csv={csv_path}")

    df = pd.read_csv(csv_path)
    _print_split_distribution(df, split_col="split")
    _print_label_distribution_all(csv_path, dataset_name=dataset_name)

    if args.verify_files:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = UNI_PathDataset(dataset_name, csv_path, "train", tokenizer, verify_files=True)
        val_ds   = UNI_PathDataset(dataset_name, csv_path, "val",   tokenizer, verify_files=True)
        test_ds  = UNI_PathDataset(dataset_name, csv_path, "test",  tokenizer, verify_files=True)

        _print_patch_stats("train", train_ds.patch_count_stats())
        _print_patch_stats("val",   val_ds.patch_count_stats())
        _print_patch_stats("test",  test_ds.patch_count_stats())

        all_counts = []
        for ds in (train_ds, val_ds, test_ds):
            all_counts.extend(ds.patch_counts or [])

        if all_counts:
            arr = np.array(all_counts, dtype=np.int32)
            overall = {
                "count": int(arr.size),
                "mean": float(arr.mean()),
                "min": int(arr.min()),
                "median": float(np.median(arr)),
                "max": int(arr.max()),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
            _print_patch_stats("OVERALL", overall)
        else:
            print("\n[OVERALL] No patch stats available.")


if __name__ == "__main__":
    main()
