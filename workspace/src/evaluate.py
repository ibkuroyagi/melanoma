#!/usr/bin/env python3

# %%
import argparse
import codecs
import logging
import os
import sys
import gc
import torch
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from torch.utils.data import DataLoader
import torchtoolbox.transform as transforms
from efficientnet_pytorch import EfficientNet
from datasets.augmentation import AdvancedHairAugmentation, DrawHair, Microscope
from datasets.mela_datasets import MelanomaDataset
from models.net import Net
from utils.seed import seed_everything
from train2 import get_arguments, init_df


# %%
def main():
    """Run training."""
    # get arguments
    args = get_arguments()
    No = args.No

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
    logging.info("Evaluation script")
    # load config
    with codecs.open(args.config, encoding="utf-8") as f:
        config = yaml.load(f, yaml.Loader)
    config.update(vars(args))
    seed_everything(config["seed"])
    fig_size = config["fig_size"]
    outdir = f"exp/{fig_size}-{No}"

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
    # setup test_loader
    test = MelanomaDataset(
        df=test_df,
        imfolder=f"../input/jpeg-melanoma-{fig_size}x{fig_size}/test/",
        train=False,
        transforms=train_transform,  # For TTA
        meta_features=meta_features,
    )
    test_loader = DataLoader(
        dataset=test,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    # setup dataloader
    oof = np.zeros((len(train_df), 1))  # Out Of Fold predictions
    preds = torch.zeros(
        (len(test), 1), dtype=torch.float32, device=device
    )  # Predictions for test test

    skf = KFold(n_splits=config["n_split"], shuffle=True, random_state=config["seed"])
    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15)), 1):
        print("=" * 20, "Fold", fold, "=" * 20)
        val_idx = train_df.loc[train_df["fold"].isin(idxV)].index
        val = MelanomaDataset(
            df=train_df.iloc[val_idx].reset_index(drop=True),
            imfolder=f"../input/jpeg-melanoma-{fig_size}x{fig_size}/train/",
            train=True,
            transforms=test_transform,
            meta_features=meta_features,
        )
        val_loader = DataLoader(
            dataset=val,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
        model_path = os.path.join(outdir, f"model_No{No}_fold{fold}_{fig_size}.pt")
        model = Net(
            arch=arch,
            n_meta_features=len(meta_features),
            in_features=config["in_features"],
        )  # New model for each fold
        model = model.to(device)
        if fold == 1:
            logging.info(model)
        model = torch.load(model_path)  # Loading best model of this fold
        model.eval()  # switch model to the evaluation mode
        val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            # Predicting on validation set once again to obtain data for OOF
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val[0] = x_val[0].to(device).float()
                x_val[1] = x_val[1].to(device).float()
                y_val = y_val.to(device).float()
                z_val = model(x_val)
                val_pred = torch.sigmoid(z_val)
                val_preds[
                    j * val_loader.batch_size : j * val_loader.batch_size
                    + x_val[0].shape[0]
                ] = val_pred
            oof[val_idx] = val_preds.cpu().numpy()

            # Predicting on test set
            for _ in range(config["TTA"]):
                for i, x_test in enumerate(test_loader):
                    x_test[0] = x_test[0].to(device).float()
                    x_test[1] = x_test[1].to(device).float()
                    z_test = model(x_test)
                    z_test = torch.sigmoid(z_test)
                    preds[
                        i * test_loader.batch_size : i * test_loader.batch_size
                        + x_test[0].shape[0]
                    ] += z_test
            preds /= config["TTA"]

        del val, val_loader, x_val, y_val, x_test
        gc.collect()
    preds /= skf.n_splits

    sns.kdeplot(pd.Series(preds.cpu().numpy().reshape(-1,)))
    plt.savefig(f"{outdir}/kde_No{No}_{fig_size}.png")
    # Saving OOF predictions so stacking would be easier
    pd.Series(oof.reshape(-1,)).to_csv(
        f"{outdir}/oof_No{No}_{fig_size}.csv", index=False
    )

    sub = pd.read_csv(
        f"../input/jpeg-melanoma-{fig_size}x{fig_size}/sample_submission.csv"
    )
    sub["target"] = preds.cpu().numpy().reshape(-1,)
    sub.to_csv(f"{outdir}/submission_No{No}_{fig_size}.csv", index=False)
    print(
        "OOF roc: {:.3f}".format(
            roc_auc_score(train_df["target"].values.astype(int), oof)
        )
    )
    print(
        "OOF acc: {:.3f}".format(
            accuracy_score(train_df["target"].values.astype(int), oof > 0.5)
        )
    )


if __name__ == "__main__":
    main()
