from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer300(nnUNetTrainer):
    """nnU-Net trainer limited to 300 epochs (default is 1000)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 300
