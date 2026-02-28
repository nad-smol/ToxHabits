# ToxHabits
## 1. Models

Trained PyTorch model checkpoints (.pt) for ToxNER and ToxUse, plus training hyperparameters. Loaded as releases.

### 1.1. ToxNER checkpoints

ToxNER checkpoints. Three files:

| File | Meaning |
|------|--------|
| **A1_BETO_toxner_opt_v1_best.pt** | **A1**: encoder BETO (`dccuchile/bert-base-spanish-wwm-cased`) + CRF. **opt_v1**: hyperparameters from the first Optuna run (15 trials) on the original train set. Used in the **Ensemble A1+A2** (ToxNER) together with the A2 checkpoint below. |
| **A2_bsc_toxner_opt_v1_best.pt** | **A2**: encoder BSC-Bio-EHR (`PlanTL-GOB-ES/bsc-bio-ehr-es`) + CRF. **opt_v1**: same Optuna optimization on original train. Main single-model (A2) reported in the paper. |
| **A2_bsc_toxner_opt_params_aug_best.pt** | Same A2 architecture and **same hyperparameters as opt_v1**, but trained on **merged train** (original + augmented data). Higher validation F1 due to more training examples. |

### 1.2. ToxUse checkpoints

ToxUse checkpoints. Same naming logic as ToxNER:

| File | Meaning |
|------|--------|
| A1_BETO_toxuse_opt_v1_best.pt | A1 (BETO+CRF), opt_v1, original train. Used in Ensemble A1+A2 (ToxUse). |
| A2_bsc_toxuse_opt_v1_best.pt | A2 (BSC-Bio-EHR+CRF), opt_v1, original train. Main single-model (A2) for ToxUse. |
| A2_bsc_toxuse_opt_params_aug_best.pt | A2, same params as opt_v1, trained on merged train (original + augmented). |



---

## 2. augmented_supplement/

Full set of augmented documents before merging with the original ToxHabits train set. Format: brat (.txt + .ann). File names include suffixes such as _aug_syn, _aug_dup1; the part before the suffix is the source document ID from the training corpus.

### 2.1. augmented_supplement/toxner/

Augmented ToxNER documents: *_aug_syn.txt and *_aug_syn.ann — synonym replacement only.

### 2.2. augmented_supplement/toxuse/

Augmented ToxUse documents: *_aug_syn.txt/.ann (synonym replacement) and *_aug_dup1.txt/.ann (additional oversampling of weak classes).

---

## 3. Root-level files

### 3.1. synonym_safe_for_augment.tsv

TSV of (term, candidate) pairs with safe_for_augment=yes and cosine similarity — the main “safe” replacement list for augmentation

### 3.2. checkpoints/hyperparameters.tsv

Tab-separated table of training hyperparameters for each checkpoint. Columns: `checkpoint_name`, `task`, `lr`, `batch_size`, `num_epochs`, `weight_decay`, `warmup_ratio`, `dropout`, `max_length`, `seed`, `train_size`, `val_size`, `val_f1`, `notes`. Source: project `output/run_log.tsv` (train phase) and `notes` for dropout where not in the log.

### 3.3. `run_predict_on_test.py`

Python script that runs a single checkpoint on the test set. It selects the correct predictor (A1 vs A2) from the checkpoint filename and calls predict_A1_BETO_improved.py or predict_A2_bsc_improved.py. Run from the project root.

Usage (examples):
python output/supplementary/run_predict_on_test.py --task toxner --checkpoint output/supplementary/checkpoints/toxner/A2_bsc_toxner_opt_v1_best.pt
python output/supplementary/run_predict_on_test.py --task toxuse --checkpoint output/supplementary/checkpoints/toxuse/A2_bsc_toxuse_opt_v1_best.pt

Optional: `--test_dir <path>`, `--out_dir <path>`
