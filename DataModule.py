import kaldi_io
import numpy
import pytorch_lightning
import torch


class XvectorDataset(torch.utils.data.Dataset):
    """Hold the x-vectors listed in a Kaldi `xvectors.scp`, as written by stklia's extract.py."""

    def __init__(self, scp_path):
        self.utterance_ids, vectors = [], []
        for utterance_id, vector in kaldi_io.read_vec_flt_scp(scp_path):
            self.utterance_ids.append(utterance_id)
            vectors.append(vector)
        self.vectors = torch.from_numpy(numpy.stack(vectors)).float()

    def __len__(self):
        return len(self.utterance_ids)

    def __getitem__(self, index):
        return self.vectors[index], self.utterance_ids[index]


class XvDataModule(pytorch_lightning.LightningDataModule):
    """Serve training and validation x-vectors for fit, and evaluation x-vectors for predict."""

    def __init__(
            self,
            train_scp: str = None,
            val_scp: str = None,
            predict_scp: str = None,
            batch_size: int = 256,
            num_workers: int = 0,
            ):
        super().__init__()
        self.train_scp = train_scp
        self.val_scp = val_scp
        self.predict_scp = predict_scp
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        if stage == "fit":
            self.training_set = XvectorDataset(self.train_scp)
            self.validation_set = XvectorDataset(self.val_scp)
        if stage == "predict":
            self.predict_set = XvectorDataset(self.predict_scp)


    def _dataloader(self, dataset, shuffle, drop_last):
        return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                drop_last=drop_last,
                pin_memory=True,
                shuffle=shuffle,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                )


    def train_dataloader(self):
        return self._dataloader(self.training_set, shuffle=True, drop_last=True)


    def val_dataloader(self):
        return self._dataloader(self.validation_set, shuffle=False, drop_last=False)


    def predict_dataloader(self):
        # Evaluation needs every utterance exactly once.
        return self._dataloader(self.predict_set, shuffle=False, drop_last=False)
