from typing import TypedDict

import torch


class CollateFnMixin:
    def collate_fn(self, batch) -> TypedDict:

        subject_id_map = [int(x[2][1:4]) - 1 for x in batch]

        waveforms = torch.stack([torch.as_tensor(x[0]) for x in batch])
        targets = torch.stack([torch.as_tensor(x[1]) for x in batch])
        global_information = torch.as_tensor(subject_id_map)

        return dict(waveform=waveforms, global_information=global_information, targets=targets)

    def collate_fn_report(self, batch):
        return self.collate_fn(batch)
