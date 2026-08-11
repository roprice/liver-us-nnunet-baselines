# Setup and training: liver US nnU-Net baselines on Verda A100

# 1. Install system dependencies
apt install python3-pip unzip python-is-python3 -y

# 2. Clone the repo
cd ~
git clone https://github.com/roprice/liver-us-nnunet-baselines.git
cd liver-us-nnunet-baselines

# 3. Fix typing_extensions conflict (known issue on Ubuntu system Python)
rm -f /usr/lib/python3/dist-packages/typing_extensions.py
rm -rf /usr/lib/python3/dist-packages/typing_extensions-*.dist-info

# 4. Install Python dependencies
pip install -r requirements.txt --break-system-packages

# 5. Set up nnU-Net directories
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# 6. Download the AUL dataset from Zenodo
cd data
pip install zenodo-get --break-system-packages
zenodo_get 7272660
unzip '*.zip' -d AUL
rm -rf AUL/__MACOSX
cd ..

# 7. Verify data structure - expect Benign/, Malignant/, Normal/
ls data/AUL/

# 8. Convert to nnU-Net format
python convert_aul.py \
  --raw-data-dir data/AUL \
  --output-dir $nnUNet_raw/Dataset001_LiverUS

# 9. Sanity check the split
ls $nnUNet_raw/Dataset001_LiverUS/imagesTr | wc -l   # expect 625
ls $nnUNet_raw/Dataset001_LiverUS/imagesTs | wc -l   # expect 110

# 10. Create subsets
python create_subsets.py \
  --source-dir $nnUNet_raw/Dataset001_LiverUS \
  --output-base $nnUNet_raw

# 11. Run training
nohup bash run_efficiency_study.sh > efficiency_study.log 2>&1 &
tail -f efficiency_study.log
