import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from trainers.base_trainer import BaseTrainer


class CNNTrainer(BaseTrainer):
    """Trainer for CNN."""

    def _train_step(self, x, y):
        # parse batch
        batch_size = x[0].size(0)
        x[0] = x[0].to(self.device).float()
        x[1] = x[1].to(self.device).float()

        # calculate loss
        z = self.model(x)
        loss = self.criterion(z, y.to(self.device).float().unsqueeze(1))
        loss = loss / batch_size

        loss.backward()
        self.forward_count += 1

        if self.forward_count == self.config["accum_grads"]:
            # update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.forward_count = 0

            # update scheduler step
            # if self.scheduler is not None:
            #     self.scheduler.step()

            # update counts
            self.steps += 1
            self._check_train_finish()

        # get and store statistics
        # preds = z.detach().cpu().numpy() > 0.5
        # labs = y.cpu().numpy()
        # self.preds = np.concatenate([self.preds, z.cpu().numpy().flatten()])
        # self.labs = np.concatenate([self.labs, labs])
        # NOTE(kan-bayashi): train/loss will be averaged over steps * accum_grads
        self.total_train_loss["train/loss"] += loss.item() * self.config["accum_grads"]
        print("train_cnn:steps", self.steps)
        # TODO(kan-bayashi): Calculation on tensor may be faster
        # self.total_train_loss["train/acc"] += accuracy_score(labs, preds)
        # self.total_train_loss["train/recall"] += recall_score(labs, preds)

    @torch.no_grad()
    def _eval_step(self, x, y):
        # parse batch
        batch_size = x[0].size(0)
        x[0] = x[0].to(self.device).float()
        x[1] = x[1].to(self.device).float()

        # calculate loss
        z = self.model(x)
        loss = self.criterion(z, y.to(self.device).float().unsqueeze(1))
        loss = loss / batch_size

        # get and store statistics
        # preds = z.cpu().numpy() > 0.5
        labs = y.cpu().numpy()
        self.preds = np.concatenate([self.preds, z.cpu().numpy().flatten()])
        self.labs = np.concatenate([self.labs, labs])
        self.total_eval_loss["eval/loss"] += loss.item()
        # TODO(kan-bayashi): Calculation on tensor may be faster
        # self.total_eval_loss["eval/acc"] += accuracy_score(labs, preds)
        # self.total_eval_loss["eval/recall"] += recall_score(labs, preds)
