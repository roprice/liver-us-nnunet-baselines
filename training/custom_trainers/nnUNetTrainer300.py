from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer300(nnUNetTrainer):
    """nnU-Net trainer limited to 300 epochs (default is 1000).

    Used for ablation experiments comparing epoch budgets.
    """
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device=None):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 300
