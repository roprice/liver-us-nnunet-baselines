# Data efficiency study of nnU-Net for malignant liver mass segmentation on B-mode ultrasound

This research tries to answer the question: what are the baseline training data needed to train nnU-Net to detect and segment malignant liver masses using B-mode ultrasound imagery?

To do this, it establishes a reproducible segmentation baseline and data efficiency analysis, using nnU-Net on the Annotated Ultrasound Liver (AUL) dataset. 

## Key findings 

The models perform well in detecting livers themselves along with malignant masses. For the largest model (625 images), liver detection rate is 100% and malignant mass detection 97%. Benign mass detection rate is 70% (21/30).

Most models also perform well at liver segmentation. The 200 training image model had Dice of 0.89 Dice, with minimal gains from tripling the data to 625 (0.90). Normal livers are the easiest (0.93), and liver boundary detection is reliable regardless of pathology type.

Mass segmentation tells a different story. Malignant mass segmentation reaches usable performance at 200 images (0.65 Dice) and continues improving to 0.77 at 625, with no sign of plateauing. 

However, benign mass segmentation remains poor across all sizes (0.40 Dice at 625), driven by two factors: smaller lesion size (benign masses are disproportionately represented in the smallest size quartile) and lower visual contrast against surrounding parenchyma, consistent with the known isoechoic presentation of common benign lesions on B-mode ultrasound (Schima et al., 2018).

Per-image analysis reveals a bimodal pattern in benign detection: the model either segments a benign mass well (Dice > 0.85) or misses it entirely (Dice = 0.0), with most such misses falling into the smallest size quartile.

Within the same size quartile, small malignant masses (0.60 Dice) still substantially outperform small benign masses (0.31 Dice), confirming that pathology type matters independently of size.

Ablation experiments at 200 images showed that neither doubling the training duration (300 epochs) nor switching to a residual encoder architecture meaningfully improved mass Dice.

For false positives on normal livers, the model is highly specific: 12 of 15 normal test cases had zero false mass predictions, and the remaining 3 had negligible counts (< 0.4% of image pixels).

Such failures may be clinically defensible. Small benign lesions are the least likely to change clinical management, while malignant mass detection, the dangerous failure mode, performs well even at small lesion sizes, even with limited training data.

## Limitations

This study uses a single dataset (AUL) from a single institution. Generalization to other ultrasound machines, patient populations, and annotation styles is untested.  The 85/15 train/test split uses a single fold rather than full cross-validation. Benign mass analysis is limited by small sample size (30 test cases) and confounded by the correlation between pathology type and lesion size in this dataset.

Checkpoint selection was based on the mean of liver and mass pseudo Dice, which may not optimize for mass detection specifically



## Results 

Training set sizes were derived by assembling nested subsets: the 25-image set is fully contained within the 50-image set, and so on; the only variable between sizes is additional data. Best checkpoint selected by EMA pseudo Dice on nnU-Net's internal validation split averaging Liver and Mass Dice (a limitation discussed under 'Limitations').

All training set sizes were evaluated on the same 110 held-out test images (15% stratified split, seed 42). 95 of 110 test cases contained annotated masses.


### Data efficiency curve

| Size | Liver Dice | Mass Dice |
|------|-----------|-----------|
| 25 | 0.820 | 0.292 |
| 50 | 0.846 | 0.358 |
| 100 | 0.878 | 0.506 |
| 200 | 0.893 | 0.546 |
| 300 | 0.898 | 0.559 |
| 400 | 0.889 | 0.587 |
| 500 | 0.899 | 0.639 |
| 625 | 0.899 | 0.652 |

### Mass segmentation by pathology type

| Size | Benign Dice | n | Malignant Dice | n |
|------|------------|---|---------------|---|
| 25 | 0.169 | 30 | 0.349 | 65 |
| 50 | 0.147 | 30 | 0.455 | 65 |
| 100 | 0.264 | 30 | 0.618 | 65 |
| 200 | 0.311 | 30 | 0.654 | 65 |
| 300 | 0.317 | 30 | 0.671 | 65 |
| 400 | 0.341 | 30 | 0.701 | 65 |
| 500 | 0.319 | 30 | 0.787 | 65 |
| 625 | 0.403 | 30 | 0.767 | 65 |

### Mass detection rates by pathology type (625 images)

A mass is counted as detected if the model predicted any mass pixels for that case.

| Category | Detected | Total | Rate |
|----------|----------|-------|------|
| Malignant | 63 | 65 | 97% |
| Benign | 21 | 30 | 70% |
| Normal (false positives) | 3 | 15 | 20% |


### Liver segmentation by pathology type (625 images)

| Category | Liver Dice | Std | n |
|----------|-----------|-----|---|
| Normal | 0.930 | 0.027 | 15 |
| Malignant | 0.901 | 0.091 | 65 |
| Benign | 0.882 | 0.108 | 30 |

### Mass Dice vs lesion size (625 images)

| Quartile | Mean Dice | Std | n | Benign | Malignant |
|----------|----------|-----|---|--------|-----------|
| Q1 (smallest) | 0.394 | 0.317 | 24 | 17 | 7 |
| Q2 | 0.572 | 0.315 | 24 | 8 | 16 |
| Q3 | 0.829 | 0.219 | 23 | 2 | 21 |
| Q4 (largest) | 0.821 | 0.128 | 24 | 3 | 21 |

Pearson correlation between mass size and Dice: r = 0.349, p = 0.0005. Size and class are confounded in this dataset: benign masses tend to be smaller, and the smallest quartile is 71% benign.


## Projected performance

Log-curve fit to the data efficiency results above (R² shown per metric).

| Size | Liver Dice | Mass Dice | Benign Mass | Malignant Mass |
|------|-----------|-----------|-------------|---------------|
| 625 (actual) | 0.899 | 0.652 | 0.403 | 0.767 |
| 1,000 (projected) | 0.919 | 0.707 | 0.407 | 0.846 |
| 2,000 (projected) | 0.936 | 0.783 | 0.456 | 0.935 |
| 5,000 (projected) | 0.957 | 0.884 | 0.520 | 0.990 |

Fit quality: mass Dice R² = 0.96, malignant mass R² = 0.95, benign mass R² = 0.89.

**Estimated data requirements for key thresholds:**

- 0.90 overall mass Dice: ~5,800 images
- 0.90 malignant mass Dice: ~1,500 images (very achievable)
- 0.70 benign mass Dice: ~63,000 images (essentially impossible through data alone)


## Training details

| Size | Peak epoch | Epoch time |
|------|-----------|------------|
| 25 | 66 | ~13s |
| 50 | 122 | ~15s |
| 100 | 148 | ~14s |
| 200 | 136 | ~21s |
| 300 | 146 | ~23s |
| 400 | 140 | ~22s |
| 500 | 147 | ~22s |
| 625 | 114 | ~20s |

Training was capped at 150 epochs. Several runs peaked near the cutoff (100 at epoch 148, 300 at 146, 500 at 147), prompting an ablation experiment to see whether more training would make a difference.


## Ablation experiments

We ran two ablation experiments - one to test whether the training was sufficient, another to test whether nnU-Net's residual encoder architecture (ResEnc M) performs better than its plain convolutional encoder, which was used for the main training runs.

### Extended training (300 epochs)

Doubling the epoch budget from 150 to 300 did not improve mass detection. Mass Dice decreased slightly from 0.546 to 0.525, suggesting mild overfitting. Liver Dice was unchanged (0.893).

### Residual encoder architecture

Replacing the plain convolutional encoder with nnU-Net's ResEnc M preset yielded a marginal improvement in mass Dice (0.556 vs 0.546) that is not statistically meaningful. Liver Dice was unchanged (0.890).

### Summary

| Configuration | Liver Dice | Mass Dice |
|---|---|---|
| PlainConvUNet, 150 epochs (baseline) | 0.893 | 0.546 |
| PlainConvUNet, 300 epochs | 0.893 | 0.525 |
| ResEnc M, 300 epochs | 0.890 | 0.556 |

Note that ResEnc M increased per-epoch training time by 70% (36s vs 21s).

Neither extended training nor a deeper encoder architecture meaningfully improved mass detection at 200 training images. This suggests that the bottleneck at this dataset size is the training data itself, not the model's capacity or training duration. 


## Preprocessing

The AUL dataset provides outline polygons marking the ultrasound scan sector boundary. Pixels outside the outline (black padding, UI overlays like the body position icon) contain no diagnostic information. During conversion, these pixels are zeroed out so that nnU-Net's automatic nonzero-mask normalization excludes non-scan-sector regions. Outline masks are also saved separately for potential use in future experiments (POCUS simulation, custom normalization).

## Model

U-Net is a family of encoder-decoder convolutional neural networks designed for biomedical image segmentation, introduced by Ronneberger et al. in 2015. This project uses nnU-Net v2 (no-new-Net), a self-configuring framework that automatically adapts U-Net architecture, preprocessing, and training hyperparameters to a given dataset. nnU-Net determines patch size, batch size, resampling strategy, and network topology from the data itself, eliminating manual tuning. For reference, see the [official repository](https://github.com/MIC-DKFZ/nnUNet).

A custom trainer (`training/custom_trainers/nnUNetTrainer150.py`) limits training to 150 epochs. The nnU-Net default is 1000 epochs.

### Labels

| Value | Label |
|-------|-------|
| 0 | Background |
| 1 | Liver |
| 2 | Mass |

## Data

The [Annotated Ultrasound Liver (AUL) dataset](https://zenodo.org/records/7272660) contains 735 B-mode liver ultrasound images across three categories: 500 malignant masses, 150 benign masses, and 85 normal (no mass). Each image has expert-annotated polygon segmentations for liver and mass regions, along with outline polygons delineating the ultrasound scan sector boundary.

Expected structure after download and extraction:

```
AUL/
  Benign/image/*.jpg
  Benign/segmentation/{liver,mass,outline}/*.json
  Malignant/image/*.jpg
  Malignant/segmentation/{liver,mass,outline}/*.json
  Normal/image/*.jpg
  Normal/segmentation/{liver,outline}/*.json
```

## Repository structure

```
training/               Training pipeline
  convert_aul.py          Convert raw AUL to nnU-Net format with outline masking
  create_subsets.py        Generate nested stratified subsets for efficiency study
  custom_trainers/         nnUNetTrainer150 (150-epoch cap), nnUNetTrainer300 (300-epoch cap)
  run_efficiency_study.sh  Train all subset sizes and run predictions
  run_ablations.sh         Epoch and architecture ablation experiments at 200 images
  train.sh                 Single-dataset training script

analysis/               Post-training evaluation and visualization
  evaluate.py              Single-run Dice evaluation
  evaluate_efficiency.py   Efficiency curve (Dice across all sizes)
  evaluate_by_class.py     Mass Dice by benign vs malignant
  dice_vs_size.py          Mass Dice vs lesion size with correlation stats
  dice_by_class_and_size.py  Size-class interaction by quartile
  normal_mass_check.py     False positives on normal cases, missed masses
  visualize_predictions.py Overlay visualizations (ground truth vs prediction)
  peak_epochs.py           Best epoch per training run
  augmentations/           nnU-Net default augmentation visualization

utilities/              Ops scripts for cloud GPU workflows
  poll_and_download.sh     Poll training server, download results
  cleanup_after_download.sh  Delete instance after download completes

data/                   Raw AUL dataset (from Zenodo)
nnUNet_raw/             Converted nnU-Net-formatted dataset
results/                Training outputs (checkpoints, predictions, logs)
```

## Setup

Requires Python 3.10+. Training requires a CUDA-capable GPU. The full efficiency study (8 training runs at 150 epochs) completes in approximately 8 hours on an A100 80GB.

Total wall-clock time including instance provisioning, setup, training, and downloading the 9GB results payload was approximately 15 hours. Evaluation was performed locally on a MacBook (no GPU required).

```bash
pip install -r requirements.txt
```

Set nnU-Net environment variables:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## Pipeline

### 1. Convert AUL to nnU-Net format

```bash
python training/convert_aul.py \
  --raw-data-dir /path/to/AUL \
  --output-dir $nnUNet_raw/Dataset001_LiverUS
```

Creates an 85/15 stratified train/test split (seed 42): 625 training images and 110 test images. Zeros out pixels outside the scan sector outline. Saves a `case_mapping.json` linking each case ID to its original category and filename.

### 2. Data efficiency study

```bash
python training/create_subsets.py \
  --source-dir $nnUNet_raw/Dataset001_LiverUS \
  --output-base $nnUNet_raw

bash training/run_efficiency_study.sh
```

Creates nested stratified subsets (25, 50, 100, 200, 300, 400, 500 images), trains all 8 sizes sequentially (including the full 625), and runs predictions on the held-out test set.

### 3. Evaluate

```bash
python analysis/evaluate_efficiency.py \
  --predictions-base $nnUNet_results \
  --labels-dir $nnUNet_raw/Dataset001_LiverUS/labelsTs

python analysis/evaluate_by_class.py
python analysis/dice_vs_size.py
```

## Training infrastructure

All training runs were performed on a single NVIDIA A100 80GB GPU (Verda Cloud, FIN-01 region). Total compute cost for the full efficiency study: approximately $14. Including instance idle time and data transfer, total cost was approximately $27..




## Future work

Improving mass detection, particularly for benign masses, will require approaches such as more annotated data, targeted augmentation strategies, annotation review, loss function weighting, adjusting the loss threshold in postprocessing, attention gating, and transfer learning. Specific next steps related to this project are:

- Mass-specific checkpoint selection: save best checkpoint based on mass Dice alone rather than the liver-mass average.
- POCUS degradation: simulate handheld probe image quality (lower resolution, increased speckle, narrower field of view) and measure segmentation robustness
- Cross-dataset validation: evaluate on SMC-LUD and other liver ultrasound datasets
- Augmentation study: systematic evaluation of US-specific augmentations (speckle noise, acoustic shadowing simulation) for mass detection

## References

- Alsharid, M., Guo, X., Men, Q. et al. (2025). On the public dissemination and open sourcing of ultrasound resources, datasets and deep learning models. npj Digit. Med. 8, 777.
- Isensee, F. et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203-211.
- Schima, W., Koh, D.M., Baron, R. (2018). Focal Liver Lesions. In: Hodler, J. et al. (eds) Diseases of the Abdomen and Pelvis 2018-2021. IDKD Springer Series. doi: 10.1007/978-3-319-75019-4_17
- Tak, J. et al. (2026). SMC-LUD: Large-Scale B-Mode Liver Ultrasound Dataset for Hepatocellular Carcinoma and Hemangioma Classification. Scientific Data.
- Tupper, A. & Gagné, C. (2025). Revisiting Data Augmentation for Ultrasound Images. TMLR. arXiv:2501.13193
- Wu, J. et al. (2024). Boundary-aware convolutional attention network for liver segmentation in ultrasound images. Scientific Reports, 14.
- Xu, Y. et al. (2022). Annotated Ultrasound Liver images dataset. Zenodo. https://zenodo.org/records/7272660

## License

Apache 2.0
