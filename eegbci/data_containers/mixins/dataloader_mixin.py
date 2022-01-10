from torch.utils.data import DataLoader


class DataloaderMixin:
    """Mixin to handle and return dataloaders for training, validation, testing and reporting scenarios."""

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def report_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn_report,
            shuffle=False,
            pin_memory=True,
        )
