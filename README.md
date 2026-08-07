# Overview

This project establishes a reproducible segmentation baseline and data efficiency analysis for liver ultrasound, built as a foundation for point-of-care ultrasound (POCUS) deployment research. It applies nnU-Net to the Annotated Ultrasound Liver (AUL) dataset and asks: how much annotated data is enough?

At full scale (625 training images), the model achieves 0.901 Dice on liver segmentation and 0.648 Dice on mass segmentation. Liver boundary detection is data-efficient, reaching 0.886 Dice with just 200 images. Mass detection never plateaus and remains data-hungry across all tested sizes, confirming it as the harder, clinically relevant challenge.

These findings establish practical guidance for deploying liver ultrasound segmentation in data-scarce environments, where a new site may have tens to hundreds of annotated images rather than thousands.

## Results

### Baseline (625 training images)

| Metric | Dice | Std |
|--------|------|-----|
| Liver segmentation | 0.901 | 0.087 |
| Mass segmentation | 0.648 | 0.343 |

### Data efficiency

| Training images | Liver Dice | Mass Dice | Peak epoch | Time to peak* | Total time*
|-----------------|-----------|-----------|------------|------------|
| 25 | 0.818 | 0.257 | 67 | 17 min | 21 min |
| 50 | 0.833 | 0.358 | 84 | 17 min | 29 min |
| 100 | 0.861 | 0.420 | 104 | 27 min | 38 min |
| 200 | 0.886 | 0.511 | 117 | 42 min | 53 min |
| 300 | 0.886 | 0.549 | 134 | 51 min | 57 min |
| 400 | 0.888 | 0.603 | 139 | 49 min | 52 min |
| 625 | 0.901 | 0.648 | 137 | 84 min | 88 min |

* Total time for 150 epochs, as trained on an 80GB-RAM A100 



Evaluated on the same 110 held-out test images across all sizes (15% stratified split, seed 42). 96 of 110 test cases contained annotated masses. Liver Dice treats both liver parenchyma and mass regions as foreground. Best checkpoint selected by EMA pseudo Dice on nnU-Net's internal validation split. Image pixels outside the annotated scan sector boundary were zeroed to exclude non-diagnostic regions from normalization.

Key findings:

- Liver segmentation plateaus around 200 images (0.886 vs 0.901 at 0.625, 1.5 points from tripling the data). A new deployment site collecting 200-300 images has a clinically useful liver boundary model.
- Mass detection gains meaningfully at every increment through 625 images and shows no sign of plateauing. More data, better augmentation, or architectural changes (e.g. attention gates) are needed to push mass Dice toward clinical utility.
- Smaller datasets peak earlier in training: 25 images exhausts its learning signal by epoch 67, while 400 images needs 139 epochs. For small deployments, fewer training epochs suffice.
- Even 25 images produces a usable liver boundary model (0.818 Dice). Mass detection at 25 images (0.257 Dice) is not usable.

## Model: nnU-Net

U-Net is a family of encoder-decoder convolutional neural networks designed for biomedical image segmentation, introduced by Ronneberger et al. in 2015. This project uses nnU-Net (no-new-Net), a self-configuring framework that automatically adapts U-Net architecture, preprocessing, and training hyperparameters to a given dataset, eliminating manual tuning and providing strong out-of-the-box performance. nnU-Net is installed via the requirements file, but for reference see the official repository: https://github.com/MIC-DKFZ/nnUNet

A custom trainer (`custom_trainers/nnUNetTrainer150.py`) limits training to 150 epochs based on observed diminishing returns. Not using it will result in the nnU-Net default of 1000 epochs.

## Data: Annotated Ultrasound Liver (AUL) from Zenodo

Download the Annotated Ultrasound Liver (AUL) dataset from Zenodo:
https://zenodo.org/records/7272660

The dataset contains 735 B-mode liver ultrasound images across three categories: 435 malignant masses, 200 benign masses, and 100 normal (no mass). Each image has expert-annotated liver and mass segmentation polygons, along with outline polygons delineating the ultrasound scan sector boundary.

We did not use outline (scan sector) masks for normalization in this study. Whether outline-informed normalization affects data efficiency is an open question we address in follow-up work.

Under "Files", select "Download all".

Extract to a local directory. The expected structure is:

```
AUL/
  Benign/image/*.jpg
  Benign/segmentation/{liver,mass,outline}/*.json
  Malignant/image/*.jpg
  Malignant/segmentation/{liver,mass,outline}/*.json
  Normal/image/*.jpg
  Normal/segmentation/{liver,outline}/*.json
```

## Training infrastructure

All training runs were performed on a single NVIDIA A100 80GB GPU 
(Verda Cloud, FIN-01 region) at $1.79/hr. Total cost for the 
efficiency study (6 runs): approximately $8.


## Setup

Requires Python 3.10+. Training requires a CUDA-capable GPU (or Apple Silicon MPS, though significantly slower). Consider renting a cloud GPU; the full data efficiency study (6 training runs) should complete in under 6 hours on an A100 with 80GB of RAM.

If `python-gdcm` fails to build (common on macOS), it can be safely ignored. This pipeline uses PNG, not DICOM.

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
python convert_aul.py \
  --raw-data-dir /path/to/AUL \
  --output-dir $nnUNet_raw/Dataset001_LiverUS
```

Creates an 85/15 stratified train/test split (seed 42): 625 training images and 110 test images.

### 2. Train baseline model

```bash
bash train.sh
```

Runs nnU-Net preprocessing, planning, and training (2D configuration, fold 0, 150 epochs). 

### 3. Data efficiency study

To reproduce the data efficiency analysis:

```bash
# Create stratified subsampled datasets (25, 50, 100, 200, 300, 400 images)
python create_subsets.py \
  --source-dir $nnUNet_raw/Dataset001_LiverUS \
  --output-base $nnUNet_raw

# Train all sizes and run inference
bash run_efficiency_study.sh
```

This trains 6 models sequentially and predicts on the same 110 test images. 

### 4. Evaluate

```bash
# Set trainer path
export nnUNet_extTrainer="$(pwd)/custom_trainers"

# Evaluate baseline
nnUNetv2_predict \
  -i $nnUNet_raw/Dataset001_LiverUS/imagesTs \
  -o predictions \
  -d 001 -c 2d -f 0 \
  -tr nnUNetTrainer150 \
  -chk checkpoint_best.pth

python evaluate.py \
  --predictions-dir predictions \
  --labels-dir $nnUNet_raw/Dataset001_LiverUS/labelsTs

# Evaluate efficiency study
python evaluate_efficiency.py \
  --results-dir $nnUNet_results \
  --labels-dir $nnUNet_raw/Dataset001_LiverUS/labelsTs
```

## Labels

| Value | Label      |
|-------|------------|
| 0     | Background |
| 1     | Liver      |
| 2     | Mass       |

## Future work

- Warm-start efficiency: pretraining on unlabeled ultrasound data (e.g. SMC-LUD) to shift the data efficiency curve leftward
- Data augmentation study: systematic evaluation of augmentation strategies for mass detection, building on Tupper & Gagné (2025) finding that most augmentations hurt mass segmentation
- POCUS degradation robustness: simulating handheld probe image quality and measuring segmentation performance loss

## References

- Asbach, P. et al. (2025). Deep learning for deep learning performance: How much data is needed for segmentation in biomedical imaging? PLOS One.
- Alsharid, M., Guo, X., Men, Q. et al. (2025). On the public dissemination and open sourcing of ultrasound resources, datasets and deep learning models. npj Digit. Med. 8, 777. https://doi.org/10.1038/s41746-025-02162-4
- Isensee, F. et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203-211.
- Syed, A.B. et al. (2023). Effect of Dataset Size and Medical Image Modality on Convolutional Neural Network Model Performance for Automated Segmentation: A CT and MR Renal Tumor Imaging Study. Journal of Digital Imaging.
- Tak, J. et al. (2026). SMC-LUD: Large-Scale B-Mode Liver Ultrasound Dataset for Hepatocellular Carcinoma and Hemangioma Classification. Scientific Data, 13, 649.
- Tupper, A. & Gagné, C. (2025). Revisiting Data Augmentation for Ultrasound Images. TMLR. arXiv:2501.13193
- Wald, T. et al. (2025). Revisiting MAE pre-training for 3D medical image segmentation. arXiv:2410.23132.
- Wu, J. et al. (2024). Boundary-aware convolutional attention network for liver segmentation in ultrasound images. Scientific Reports, 14.
- Xu, Y. et al. (2023). Annotated Ultrasound Liver images dataset. Zenodo. https://zenodo.org/records/7272660
- Zhou, Y. et al. (2021). Annotation-efficient deep learning for automatic medical image segmentation. Nature Communications.
- Zhuang, L. et al. (2025). Advancing Precision Oncology Through Modeling of Longitudinal and Multimodal Data. arXiv:2502.07836.

## License

Apache 2.0
