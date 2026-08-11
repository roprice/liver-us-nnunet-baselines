import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer300(nnUNetTrainer):
    """nnU-Net trainer limited to 300 epochs.

    Used for ablation experiments comparing epoch budgets
    against the 150-epoch baseline.
    """
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300
