import logging
import os

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from tqdm import tqdm


class BaseTrainer(ABC):
    """Base trainer module."""

    def __init__(
        self,
        steps: int,
        epochs: int,
        config: Dict,
        train_data_loader: torch.utils.data.DataLoader,
        valid_data_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize trainer.
        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            train_data_loader (torch.utils.data.Dataloader): Training data loader.
            valid_data_loader (torch.utils.data.Dataloader): Validation data loader.
            model (torch.nn.Module): Model instance.
            optimizer (torch.optim.Optimizer): Optimizer instance.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler instance.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.
        """
        self.steps = steps
        self.epochs = epochs
        self.config = config
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.criterion = nn.BCEWithLogitsLoss()
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.forward_count = 0
        self.best_val = np.inf
        self.preds = np.empty(0)
        self.labs = np.empty(0)
        self.val_roc = 0

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps, total=self.config["steps"], desc="[train]")
        while True:
            # train one epoch
            self._train_epoch()
            self._eval_epoch()
            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "steps": self.steps,
            "epochs": self.epochs,
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)
        pass

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state_dict["scheduler"])

    @abstractmethod
    def _train_step(self, x, y):
        """Train model one step."""
        # write training procedure here

        # 1. calculate loss and update parameters

        # 2. store loss value in total_train_loss dict
        # E.g.: self.total_train_loss["train/loss"] += loss.item()

        # 3. update counts
        self.steps += 1

        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, (x, y) in enumerate(self.train_data_loader, 1):
            # train one step
            self._train_step(x, y)
            self.tqdm.update(1)
            # check interval
            print("check interval", self.config["rank"], self.forward_count)
            if self.config["rank"] == 0 and self.forward_count == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()
                print("save results")
            # break
            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.train_steps_per_epoch = train_steps_per_epoch // self.config["accum_grads"]
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

    @abstractmethod
    @torch.no_grad()
    def _eval_step(self, x, y):
        """Evaluate model one step."""
        # write evaluation procedure here

        # 1. calculate loss value and store it in total_eval_loss dict
        # E.g.: self.total_eval_loss["eval/loss"] += loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        self.model.eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, (x, y) in enumerate(
            tqdm(self.valid_data_loader, desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(x, y)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )
        self.val_roc = roc_auc_score(self.labs.astype(np.int64), self.preds)
        self.total_eval_loss["eval/roc"] = self.val_roc
        if self.val_roc >= self.best_val:
            self.best_val = self.val_roc
            # patience = es_patience  # Resetting patience since we have new best validation accuracy
            self.save_checkpoint(
                os.path.join(self.config["outdir"], self.config["model_path"])
            )
            logging.info(
                f"Successfully saved at {os.path.join(self.config['outdir'], self.config['model_path'])}."
            )
        self.scheduler.step(self.val_roc)
        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)
        self.preds = np.empty(0)
        self.labs = np.empty(0)

        # restore mode
        self.model.train()

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pt")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")
            print(os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pt"))

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= (
                    self.config["log_interval_steps"] * self.config["accum_grads"]
                )
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["steps"]:
            self.finish_train = True
