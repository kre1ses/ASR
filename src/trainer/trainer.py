from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer

from torch.cuda.amp import GradScaler, autocast
from accelerate import Accelerator
import torch


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch_idx, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            if (batch_idx + 1) % 4 == 0:
                self.optimizer.zero_grad()
        
        # if self.use_accelerate:
        #     autocast_ctx = self.accelerator.autocast
        # else:
        #     autocast_ctx = torch.cuda.amp.autocast
        if self.is_train:
            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                batch.update(outputs)

        # with autocast_ctx():
                all_losses = self.criterion(**batch)
                batch.update(all_losses)

        # if self.is_train:
            # if self.use_accelerate:
                self.accelerator.backward(batch["loss"])
                self._clip_grad_norm()

                if (batch_idx + 1) % 4 == 0:
                    self.optimizer.step()

                self.lr_scheduler.step()
        else:
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

            # else:
            #     self.scaler.scale(batch["loss"]).backward()
            #     self._clip_grad_norm()

            #     if (batch_idx + 1) % 4 == 0:
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()

            #     self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        if self.is_train:
            log_step = 10
        else:
            log_step = 1
        if (batch_idx + 1) % log_step == 0:
            for loss_name in self.config.writer.loss_names:
                metrics.update(loss_name, batch[loss_name].item())
        if (batch_idx + 1) % log_step == 0:
            for met in metric_funcs:
                metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example
        if not self.accelerator.is_main_process:
            return

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.writer.add_audio(
                "train_audio", batch["audio"][0], sample_rate=16000
                )
            
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot.transpose(-2,-1))
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        argmax_texts = [elem.lower().replace("'", "").replace("[unk]", "").strip() for elem in argmax_texts]

        if self.text_encoder.beam_use:
            beam_texts = []

            if self.text_encoder.lm_use:
                predictions = log_probs.detach().cpu()
                lengths = log_probs_length.detach()
                beam_texts = self.text_encoder.ctc_lm_beam_search(predictions, lengths) ############
            else:
                predictions = log_probs.detach().cpu().numpy()
                lengths = log_probs_length.detach().numpy()
                for log_prob_vec, length in zip(predictions, lengths):
                    beams = self.text_encoder.ctc_beam_search(log_prob_vec[:length])[0][0]
                    beams = beams.lower().replace("'", "").replace("[unk]", "").strip()
                    beam_texts.append(beams)
            tuples = list(zip(beam_texts, argmax_texts, text, argmax_texts_raw, audio_path))

            rows = {}
            for beam_pred, pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
                target = self.text_encoder.normalize_text(target)
                wer = calc_wer(target, pred)
                cer = calc_cer(target, pred)

                beam_wer = calc_wer(target, beam_pred)
                beam_cer = calc_cer(target, beam_pred)

                rows[Path(audio_path).name] = {
                    "target": target,
                    "raw prediction": raw_pred,
                    "predictions": pred,
                    "wer": wer,
                    "cer": cer,
                    "beam_search_predictions": beam_pred,
                    "beam_search_wer": beam_wer,
                    "beam_search_cer": beam_cer,
                }

        else:
            tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))

            rows = {}
            for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
                target = self.text_encoder.normalize_text(target)
                wer = calc_wer(target, pred) * 100
                cer = calc_cer(target, pred) * 100

                rows[Path(audio_path).name] = {
                    "target": target,
                    "raw prediction": raw_pred,
                    "predictions": pred,
                    "wer": wer,
                    "cer": cer,
                }
                
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
