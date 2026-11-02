import pickle

import numpy
import pytorch_lightning
import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class SPINE(pytorch_lightning.LightningModule):
    """Autoencoder that maps pretrained x-vectors to sparse embeddings."""

    def __init__(
            self,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            sparsity: float, # Sets the average sparsity loss threshold: rho_star = 1 - sparsity
            std_noise: float, # The std of the gaussian noise
            r_lambda: float,
            as_lambda: float,
            ps_lambda: float,
            binarize: bool = False, # Output binary embeddings from encode() and predict_step()
            binarization_threshold: float = 0.5,
            ):

        super(SPINE, self).__init__()

        # log hyperparameters
        self.save_hyperparameters()


        self.add_module("encoder", module=encoder)
        self.add_module("decoder", module=decoder)

        self.std_noise = std_noise

        self.RLoss = self._getRLoss
        self.ASLoss = self._getASLoss
        self.PSLoss = self._getPSLoss

        self.r_lambda = r_lambda
        self.as_lambda = as_lambda
        self.ps_lambda = ps_lambda

        self.sparsity = sparsity
        self.rho_star = 1 - self.sparsity

        self.binarize = binarize
        self.binarization_threshold = binarization_threshold

    def _getASLoss(self, hidden):
        temp = torch.mean(hidden, dim=0) - self.rho_star
        temp = temp.clamp(min=0)
        return torch.mean(temp**2) * self.as_lambda

    def _getPSLoss(self, hidden):
        return torch.mean(hidden * (1-hidden)) * self.ps_lambda


    def _getRLoss(self, reconstructed_vector, in_vector):
        return torch.nn.MSELoss()(reconstructed_vector, in_vector) * self.r_lambda


    def encode(self, in_vector):
        """Return the SPINE embedding, thresholded when the model is set to binarize."""
        spine_vector = self.encoder(in_vector).clamp(0, 1)

        if self.binarize:
            spine_vector = (spine_vector >= self.binarization_threshold).to(spine_vector.dtype)

        return spine_vector


    def forward(self, in_vector):
        """Return continuous embeddings and their reconstruction; training never binarizes."""
        spine_vector = self.encoder(in_vector).clamp(0, 1)
        reconstructed_vector = self.decoder(spine_vector)

        return spine_vector, reconstructed_vector


    def training_step(self, batch, batch_idx):

        in_vector = batch[0]
        noised_in_vector = in_vector + torch.normal(0, self.std_noise, size=in_vector.shape).to(in_vector.device)

        spine_vector, reconstructed_vector = self(in_vector)

        r_loss = self.RLoss(reconstructed_vector, noised_in_vector)
        as_loss = self.ASLoss(spine_vector)
        ps_loss = self.PSLoss(spine_vector)

        self.log("train_r_loss", r_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("train_as_loss", as_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("train_ps_loss", ps_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return r_loss + as_loss + ps_loss


    def validation_step(self, batch, batch_idx):

        in_vector = batch[0]
        noised_in_vector = in_vector + torch.normal(0, self.std_noise, size=in_vector.shape).to(in_vector.device)

        spine_vector, reconstructed_vector = self(in_vector)

        r_loss = self.RLoss(reconstructed_vector, noised_in_vector)
        as_loss = self.ASLoss(spine_vector)
        ps_loss = self.PSLoss(spine_vector)

        self.log("val_r_loss", r_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("val_as_loss", as_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("val_ps_loss", ps_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        self.log("sparsity", len(spine_vector.flatten()[(spine_vector.flatten() == 0)])/len(spine_vector.flatten()), on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return r_loss + as_loss + ps_loss


    def predict_step(self, batch, batch_idx):

        return {
                "embedding": self.encode(batch[0]).cpu().numpy(),
                "label": list(batch[1]),
                }


class EmbeddingWriter(BasePredictionWriter):
    """Save predicted embeddings to a pickle with `embedding` and `label` keys."""

    def __init__(self, output_path: str):
        super().__init__(write_interval="epoch")
        self.output_path = output_path

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        batches = predictions[0] if isinstance(predictions[0], list) else predictions

        with open(self.output_path, "wb") as f:
            pickle.dump({
                "embedding": numpy.vstack([batch["embedding"] for batch in batches]),
                "label": [label for batch in batches for label in batch["label"]],
                }, f)
