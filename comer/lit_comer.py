import warnings
import zipfile
from typing import List

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor
from timm.scheduler import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler

from comer.datamodule import Batch, vocab
from comer.model.comer import CoMER
from comer.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,
                               to_bi_tgt_out)


class LitCoMER(pl.LightningModule):
    def __init__(
            self,
            d_model: int,
            # encoder
            growth_rate: int,
            num_layers: int,
            # decoder
            nhead: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            attn_dropout: float,
            dc: int,
            use_moe: bool,
            num_experts: int,
            qk_norm: bool,
            cross_coverage: bool,
            self_coverage: bool,
            # beam search
            beam_size: int,
            max_len: int,
            alpha: float,
            early_stopping: bool,
            temperature: float,
            # training
            learning_rate: float,
            patience: int,
            group_decay: bool,
            l_aux_weight: float,
            perturb_mode: str,
            perturb_prob: float,

    ):
        super().__init__()
        self.save_hyperparameters()

        # self.comer_model = None
        self.comer_model = CoMER(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            use_moe=use_moe,
            num_experts=num_experts,
            qk_norm=qk_norm,
            dropout=dropout,
            attn_dropout=attn_dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.exprate_recorder = ExpRateRecorder()
        if perturb_mode != "none" and perturb_prob == 0.0:
            warnings.warn(
                f"perturb_mode='{perturb_mode}' but perturb_prob={perturb_prob}. "
                f"Please set perturb_prob > 0 to enable perturbation. ",
                stacklevel=2,
            )

    def setup(self, stage=None):
        if self.comer_model is None:
            warnings.warn("Model is not initialized. Build models dynamically...", stacklevel=2)
            self.comer_model = CoMER(
                d_model=self.hparams.d_model,
                growth_rate=self.hparams.growth_rate,
                num_layers=self.hparams.num_layers,
                nhead=self.hparams.nhead,
                num_decoder_layers=self.hparams.num_decoder_layers,
                dim_feedforward=self.hparams.dim_feedforward,
                use_moe=self.hparams.use_moe,
                num_experts=self.hparams.num_experts,
                qk_norm=self.hparams.qk_norm,
                dropout=self.hparams.dropout,
                attn_dropout=self.hparams.attn_dropout,
                dc=self.hparams.dc,
                cross_coverage=self.hparams.cross_coverage,
                self_coverage=self.hparams.self_coverage,
            )

    def forward(
            self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.comer_model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device, self.hparams.perturb_mode, self.hparams.perturb_prob)
        output = self(batch.imgs, batch.mask, tgt)
        out_hat, l_aux = output[0], output[1]

        if self.hparams.use_moe and l_aux is not None:
            loss = ce_loss(out_hat, out)
            l_aux = l_aux * self.hparams.l_aux_weight
            total_loss = loss + l_aux
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_l_aux", l_aux, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            return total_loss
        else:
            loss = ce_loss(out_hat, out)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss

    @torch.inference_mode()
    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device, "none", 0)
        output = self(batch.imgs, batch.mask, tgt)
        out_hat, l_aux = output[0], output[1]

        if self.hparams.use_moe and l_aux is not None:
            loss = ce_loss(out_hat, out)
            l_aux = l_aux * self.hparams.l_aux_weight
            total_loss = loss + l_aux
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_l_aux", l_aux, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            loss = ce_loss(out_hat, out)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    @torch.inference_mode()
    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps]

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)

    def approximate_joint_search(
            self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.comer_model.beam_search(img, mask, **self.hparams)

    def lr_scheduler_step(self, scheduler, metric):
        if isinstance(scheduler, CosineLRScheduler):
            scheduler.step_update(self.global_step)
        elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters() if not self.hparams.group_decay else self._decay_group(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _decay_group(self):
        if self.trainer.is_global_zero:
            print("Grouping parameters for weight decay...")

        cnn_decay = set()
        linear_decay = set()
        no_decay = set()

        cnn_modules = (torch.nn.Conv2d,)
        linear_modules = (torch.nn.Linear,)

        blacklist_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.GroupNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):

                fpn = f"{mn}.{pn}" if mn else pn

                # bias → no decay
                if pn.endswith("bias") or pn.endswith("in_proj_bias"):
                    no_decay.add(fpn)

                # norm / embedding → no decay
                elif pn.endswith("weight") and isinstance(m, blacklist_modules):
                    no_decay.add(fpn)

                # CNN weight
                elif pn.endswith("weight") and isinstance(m, cnn_modules):
                    cnn_decay.add(fpn)

                # Linear weight
                elif (pn.endswith("weight") and isinstance(m, linear_modules)) or pn.endswith("in_proj_weight"):
                    linear_decay.add(fpn)

                else:
                    warnings.warn(f"Unrecognized module {mn} in {pn}, ")

        param_dict = {pn: p for pn, p in self.named_parameters()}

        inter_params = (cnn_decay | linear_decay) & no_decay
        assert len(inter_params) == 0

        union_params = cnn_decay | linear_decay | no_decay
        assert len(param_dict.keys() - union_params) == 0
        if self.trainer.is_global_zero:
            print("Finish grouping parameters for weight decay.")

        return [
            {
                "params": [param_dict[pn] for pn in sorted(cnn_decay)],
                "weight_decay": 1e-4,
            },
            {
                "params": [param_dict[pn] for pn in sorted(linear_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]