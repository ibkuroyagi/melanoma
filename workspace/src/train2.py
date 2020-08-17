#!/usr/bin/env python3

# %%
import argparse
import codecs
import logging
import os
import sys
import gc
import time
import datetime
import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtoolbox.transform as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

from datasets.augmentation import AdvancedHairAugmentation, DrawHair, Microscope
from datasets.mela_datasets import MelanomaDataset
from models.net import Net
from utils.seed import seed_everything


def get_arguments():
    """Get arguments."""
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument("--No", type=int, default=0, help="hte number of experiments.")
    parser.add_argument(
        "--pass_list", nargs="+", type=int, default=[10], help="pass fold."
    )
    parser.add_argument(
        "--fig_size", type=int, default=0, help="hte number of experiments."
    )
    parser.add_argument("--TTA", type=int, default=3, help="the number of TTA.")
    parser.add_argument(
        "--resume", type=str, nargs="?", help="checkpoint path to resume."
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Logging level. Higher is more logging. (default=1)",
    )
    return parser.parse_args()


def init_df(
    train_path="../input/jpeg-melanoma-256x256/train.csv",
    test_path="../input/jpeg-melanoma-256x256/test.csv",
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # Using triple stratified KFolds
    # tmp = pd.read_csv("../input/melanoma-256x256/train.csv")
    train_df["fold"] = train_df["tfrecord"].copy()
    # del tmp
    # gc.collect()

    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat(
        [
            train_df["anatom_site_general_challenge"],
            test_df["anatom_site_general_challenge"],
        ],
        ignore_index=True,
    )
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix="site")
    train_df = pd.concat([train_df, dummies.iloc[: train_df.shape[0]]], axis=1)
    test_df = pd.concat(
        [test_df, dummies.iloc[train_df.shape[0] :].reset_index(drop=True)], axis=1
    )

    # Sex features
    train_df["sex"] = train_df["sex"].map({"male": 1, "female": 0})
    test_df["sex"] = test_df["sex"].map({"male": 1, "female": 0})
    train_df["sex"] = train_df["sex"].fillna(-1)
    test_df["sex"] = test_df["sex"].fillna(-1)

    # Age features
    train_df["age_approx"] /= train_df["age_approx"].max()
    test_df["age_approx"] /= test_df["age_approx"].max()
    train_df["age_approx"] = train_df["age_approx"].fillna(0)
    test_df["age_approx"] = test_df["age_approx"].fillna(0)

    train_df["patient_id"] = train_df["patient_id"].fillna(0)
    meta_features = ["sex", "age_approx"] + [
        col for col in train_df.columns if "site_" in col
    ]
    meta_features.remove("anatom_site_general_challenge")
    return (train_df, test_df, meta_features)


def main():
    """Run training."""
    # get arguments
    args = get_arguments()
    No = args.No
    fig_size = args.fig_size
    pass_list = args.pass_list
    print(f"pass_list:{pass_list}")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")
    logging.info("Training script")
    # load config
    with codecs.open(args.config, encoding="utf-8") as f:
        config = yaml.load(f, yaml.Loader)
    config.update(vars(args))
    seed_everything(config["seed"])

    # save config
    outdir = f"exp/{fig_size}-{No}"
    os.makedirs(outdir, exist_ok=True)

    with codecs.open(
        f"{outdir}/config{fig_size}-{No}.yaml", "w", encoding="utf-8"
    ) as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)

    # check config
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    train_transform = transforms.Compose(
        [
            AdvancedHairAugmentation(hairs_folder="../input/melanoma-hairs"),
            transforms.RandomResizedCrop(size=fig_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Microscope(p=config["microscope"]),
            DrawHair(hairs=config["hair"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    arch = EfficientNet.from_pretrained(config["pretrained"])

    train_df, test_df, meta_features = init_df(
        train_path=f"../input/jpeg-melanoma-{fig_size}x{fig_size}/train.csv",
        test_path=f"../input/jpeg-melanoma-{fig_size}x{fig_size}/test.csv",
    )
    print("meta_features:", len(meta_features))
    columns = [
        "epoch",
        "loss",
        "loss_val",
        "train_acc",
        "val_acc",
        "train_roc",
        "val_roc",
        "lr",
    ]
    # setup dataloader
    skf = KFold(n_splits=config["n_split"], shuffle=True, random_state=config["seed"])
    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15)), 1):
        if fold in pass_list:
            print(f"fold:{fold} was passed.")
            continue
        print("=" * 20, "Fold", fold, "=" * 20)

        train_idx = train_df.loc[train_df["fold"].isin(idxT)].index
        val_idx = train_df.loc[train_df["fold"].isin(idxV)].index

        train = MelanomaDataset(
            df=train_df.iloc[train_idx].reset_index(drop=True),
            imfolder=f"../input/jpeg-melanoma-{fig_size}x{fig_size}/train/",
            train=True,
            transforms=train_transform,
            meta_features=meta_features,
        )
        val = MelanomaDataset(
            df=train_df.iloc[val_idx].reset_index(drop=True),
            imfolder=f"../input/jpeg-melanoma-{fig_size}x{fig_size}/train/",
            train=True,
            transforms=test_transform,
            meta_features=meta_features,
        )

        train_loader = DataLoader(
            dataset=train,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset=val,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
        # New model for each fold
        model = Net(
            arch=arch,
            n_meta_features=len(meta_features),
            in_features=config["in_features"],
        )
        model = model.to(device)
        if fold == 1:
            logging.info(model)
        criterion = nn.BCEWithLogitsLoss()

        # setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            patience=config["patience"],
            verbose=True,
            factor=config["factor"],
        )
        best_val = 0
        model_path = os.path.join(outdir, f"model_No{No}_fold{fold}_{fig_size}.pt")
        log_path = os.path.join(outdir, f"log_No{No}_fold{fold}_{fig_size}.csv")
        fig_path = os.path.join(outdir, f"fig_No{No}_fold{fold}_{fig_size}.png")
        print(f"model_path:{model_path}")
        print(f"log_path:{log_path}")
        print(f"fig_path:{fig_path}")
        log_df = pd.DataFrame(np.empty((0, 8)), columns=columns)
        for epoch in range(config["epoch"]):
            start_time = time.time()
            correct = 0
            epoch_loss = 0
            epoch_loss_val = 0
            model.train()
            train_preds = torch.zeros(
                (len(train_idx), 1), dtype=torch.float32, device=device
            )
            train_true = torch.zeros(
                (len(train_idx), 1), dtype=torch.float32, device=device
            )
            for j, (x, y) in enumerate(tqdm(train_loader)):
                x[0] = x[0].to(device).float()
                x[1] = x[1].to(device).float()
                optimizer.zero_grad()
                z = model(x)
                loss = criterion(z, y.to(device).float().unsqueeze(1))
                loss.backward()
                optimizer.step()
                # round off sigmoid to obtain predictions
                pred = torch.round(torch.sigmoid(z))
                train_preds[
                    j * train_loader.batch_size : j * train_loader.batch_size
                    + x[0].shape[0]
                ] = pred
                train_true[
                    j * train_loader.batch_size : j * train_loader.batch_size
                    + x[0].shape[0]
                ] = y.float().unsqueeze(1)
                # tracking number of correctly predicted samples
                correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()
                epoch_loss += loss.item()
            epoch_loss /= len(train_idx)
            train_acc = correct / len(train_idx)
            train_roc = roc_auc_score(
                train_true.cpu().detach().numpy(), train_preds.cpu().detach().numpy()
            )
            # switch model to the evaluation mode
            model.eval()
            val_preds = torch.zeros(
                (len(val_idx), 1), dtype=torch.float32, device=device
            )
            with torch.no_grad():  # Do not calculate gradient since we are only predicting
                # Predicting on validation set
                for j, (x_val, y_val) in enumerate(tqdm(val_loader)):
                    x_val[0] = x_val[0].to(device).float()
                    x_val[1] = x_val[1].to(device).float()
                    y_val = y_val.to(device).float()
                    z_val = model(x_val)
                    val_pred = torch.sigmoid(z_val)
                    val_preds[
                        j * val_loader.batch_size : j * val_loader.batch_size
                        + x_val[0].shape[0]
                    ] = val_pred
                    loss = criterion(z_val, y_val.to(device).float().unsqueeze(1))
                    epoch_loss_val += loss.item()
                epoch_loss_val /= len(val_idx)
                val_acc = accuracy_score(
                    train_df.iloc[val_idx]["target"].values,
                    torch.round(val_preds.cpu()),
                )
                val_roc = roc_auc_score(
                    train_df.iloc[val_idx]["target"].values, val_preds.cpu()
                )

                print(
                    "Epoch {:03}: | Loss: {:.4f} | val Loss: {:.4f}|Train acc: {:.4f} | Val acc: {:.4f} | Train roc_auc: {:.4f} | Val roc_auc: {:.4f} | lr: {:.5f} | Training time: {}".format(
                        epoch + 1,
                        epoch_loss,
                        epoch_loss_val,
                        train_acc,
                        val_acc,
                        train_roc,
                        val_roc,
                        optimizer.param_groups[0]["lr"],
                        str(datetime.timedelta(seconds=time.time() - start_time))[:7],
                    )
                )
                tmp_df = pd.DataFrame(
                    [
                        [
                            epoch,
                            epoch_loss,
                            epoch_loss_val,
                            train_acc,
                            val_acc,
                            train_roc,
                            val_roc,
                            optimizer.param_groups[0]["lr"],
                        ]
                    ],
                    columns=columns,
                )
                log_df = pd.concat([log_df, tmp_df], axis=0)
                log_df.to_csv(log_path, index=False)
                plt.figure(figsize=(10, 10))
                plt.subplot(2, 2, 1)
                plt.plot(
                    log_df.loc[:, columns[0]],
                    log_df.loc[:, columns[1]],
                    label=columns[1],
                )
                plt.plot(
                    log_df.loc[:, columns[0]],
                    log_df.loc[:, columns[2]],
                    label=columns[2],
                )
                plt.legend()
                plt.subplot(2, 2, 2)
                plt.plot(
                    log_df.loc[:, columns[0]],
                    log_df.loc[:, columns[3]],
                    label=columns[3],
                )
                plt.plot(
                    log_df.loc[:, columns[0]],
                    log_df.loc[:, columns[4]],
                    label=columns[4],
                )
                plt.legend()
                plt.subplot(2, 2, 3)
                plt.plot(
                    log_df.loc[:, columns[0]],
                    log_df.loc[:, columns[5]],
                    label=columns[5],
                )
                plt.plot(
                    log_df.loc[:, columns[0]],
                    log_df.loc[:, columns[6]],
                    label=columns[6],
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.clf()
                plt.close()
                scheduler.step(val_roc)

                if val_roc >= best_val:
                    best_val = val_roc
                    patience = config[
                        "es_patience"
                    ]  # Resetting patience since we have new best validation accuracy
                    torch.save(model, model_path)  # Saving current best model
                    print(f"Score is updated {best_val:.4f}, saved@{model_path}")
                else:
                    patience -= 1
                    if patience == 0:
                        print(
                            "Early stopping. Best Val roc_auc: {:.4f}".format(best_val)
                        )
                        break
        del train, val, train_loader, val_loader, x, y, x_val, y_val
        gc.collect()


if __name__ == "__main__":
    main()
