# TaxoMIL: Taxonomy-Constrained Learning for Hierarchical Whole Slide Image Analysis [ECCV 2026]

Official implementation of **TaxoMIL: Taxonomy-Constrained Learning for Hierarchical Whole Slide Image Analysis**.

<img width="2957" height="1241" alt="pipeline" src="https://github.com/user-attachments/assets/c01c9a18-9121-45a6-91e7-1f53aa9f04d2" />


## Framework

TaxoMIL consists of three main components:

1. A MIL backbone that aggregates patch-level WSI features into a slide-level representation.
2. A dual-branch conditioning module for coarse-level and fine-level diagnosis.
3. A dual-head text decoder that generates hierarchical diagnostic labels.

The framework is trained with text generation loss and taxonomy-guided auxiliary objectives.

## Installation

```bash
git clone https://github.com/chaey0/TaxoMIL.git
cd TaxoMIL

conda create -n taxomil python=3.10 -y
conda activate taxomil

pip install -r requirements.txt
```

## Data Preparation

TaxoMIL uses pre-extracted WSI feature bags as input.

Each CSV file should contain the following fields:

```text
split,coarse_label,fine_label,UNI_features_path
```

Example:

```text
train,Benign,Usual Ductal Hyperplasia,/path/to/features/sample_001.pt
test,Malignant,Invasive Carcinoma,/path/to/features/sample_002.pt
```

Label definitions and taxonomy mappings are provided in:

```text
config/
```

The expected feature format is a `.pt` file containing patch-level embeddings for each WSI.

## Running Experiments

### Training

Train the base model:

```bash
python main.py \
  --mode train \
  --data BRACS \
  --decoder_type GPT2
```

Train TaxoMIL with all taxonomy-guided objectives:

```bash
python main.py \
  --mode train \
  --data BRACS \
  --decoder_type GPT2 \
  --use_all_loss
```

### Testing

```bash
python main.py \
  --mode test \
  --data BRACS \
  --ckpt_path checkpoints/BRACS/model.pth
```

## Repository Structure

```text
TaxoMIL/
├── main.py
├── decoder.py
├── aggregator/
├── dataloader/
├── trainer/
├── utils/
├── config/
└── assets/
```

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{taxomil2026,
  title     = {TaxoMIL: Taxonomy-Constrained Learning for Hierarchical Whole Slide Image Analysis},
  author    = {Lee, Chaeyeon and Nguyen Quoc, Khang and Song, Jinsol and Chong, Yosep and Yim, Kwangil and others},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```
