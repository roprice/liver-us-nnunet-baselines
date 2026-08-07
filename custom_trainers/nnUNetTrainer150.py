import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer150(nnUNetTrainer):
    """nnU-Net trainer limited to 150 epochs.

    The default nnU-Net trainer runs 1000 epochs. For 2D liver ultrasound
    on the AUL dataset, convergence occurs well before 150 epochs across
    all tested training set sizes (25-625 images).
    """
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
