import pytorch_lightning as pl
import torch


class BaseDataModule(pl.LightningDataModule):
    """
    Base class for all datamodules to inherit from.
    """

    def __init__(self):
        super().__init__()

    @property
    def sampling_frequency(self):
        assert self._sampling_frequency is not None, f"Specify sampling_frequency property for the DataModule ({self})"
        return self._sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, sampling_frequency: int) -> int:
        self._sampling_frequency = sampling_frequency

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        """Default way to transfer batch to device, is iterating through dict keys and moving tensors."""
        # NOTE: This won't work for nested dicts
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
