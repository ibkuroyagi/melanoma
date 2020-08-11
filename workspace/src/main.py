#!usr/bin/env python3
# %%

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pandas as pd
import numpy as np
import gc
import os
import cv2
import time
import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from efficientnet_pytorch import EfficientNet

from datasets.augmentation import AdvancedHairAugmentation, DrawHair, Microscope
from datasets.mela_datasets import MelanomaDataset
from models.net import Net
from utils.seed import seed_everything

warnings.simplefilter("ignore")


seed_everything(108)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

train_transform = transforms.Compose(
    [
        AdvancedHairAugmentation(hairs_folder="../input/melanoma-hairs"),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Microscope(p=0.5),
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


# %%
arch = EfficientNet.from_pretrained("efficientnet-b1")
train_df = pd.read_csv("../input/jpeg-melanoma-256x256/train.csv")
test_df = pd.read_csv("../input/jpeg-melanoma-256x256/test.csv")
# Using triple stratified KFolds
# tmp = pd.read_csv("../input/melanoma-256x256/train.csv")
train_df["fold"] = train_df["tfrecord"]
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


# %%
meta_features = ["sex", "age_approx"] + [
    col for col in train_df.columns if "site_" in col
]
meta_features.remove("anatom_site_general_challenge")
test = MelanomaDataset(
    df=test_df,
    imfolder="../input/jpeg-melanoma-256x256/test/",
    train=False,
    transforms=train_transform,  # For TTA
    meta_features=meta_features,
)


# %%
epochs = 15  # Number of epochs to run
es_patience = (
    3  # Early Stopping patience - for how many epochs with no improvements to wait
)
TTA = 3  # Test Time Augmentation rounds

oof = np.zeros((len(train_df), 1))  # Out Of Fold predictions
preds = torch.zeros(
    (len(test), 1), dtype=torch.float32, device=device
)  # Predictions for test test

skf = KFold(n_splits=5, shuffle=True, random_state=47)
for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15)), 1):
    print("=" * 20, "Fold", fold, "=" * 20)

    train_idx = train_df.loc[train_df["fold"].isin(idxT)].index
    val_idx = train_df.loc[train_df["fold"].isin(idxV)].index

    model_path = f"model_{fold}.pth"  # Path and filename to save model to
    best_val = 0  # Best validation score within this fold
    patience = es_patience  # Current patience counter
    arch = EfficientNet.from_pretrained("efficientnet-b1")
    model = Net(
        arch=arch, n_meta_features=len(meta_features)
    )  # New model for each fold
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer=optim, mode="max", patience=1, verbose=True, factor=0.2
    )
    criterion = nn.BCEWithLogitsLoss()

    train = MelanomaDataset(
        df=train_df.iloc[train_idx].reset_index(drop=True),
        imfolder="../input/jpeg-melanoma-256x256/train/",
        train=True,
        transforms=train_transform,
        meta_features=meta_features,
    )
    val = MelanomaDataset(
        df=train_df.iloc[val_idx].reset_index(drop=True),
        imfolder="../input/jpeg-melanoma-256x256/train/",
        train=True,
        transforms=test_transform,
        meta_features=meta_features,
    )

    train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test, batch_size=16, shuffle=False, num_workers=2)

    for epoch in range(epochs):
        start_time = time.time()
        correct = 0
        epoch_loss = 0
        model.train()

        for x, y in train_loader:
            x[0] = torch.tensor(x[0], device=device, dtype=torch.float32)
            x[1] = torch.tensor(x[1], device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            optim.zero_grad()
            z = model(x)
            loss = criterion(z, y.unsqueeze(1))
            loss.backward()
            optim.step()
            pred = torch.round(
                torch.sigmoid(z)
            )  # round off sigmoid to obtain predictions
            correct += (
                (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()
            )  # tracking number of correctly predicted samples
            epoch_loss += loss.item()
        train_acc = correct / len(train_idx)

        model.eval()  # switch model to the evaluation mode
        val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
        with torch.no_grad():  # Do not calculate gradient since we are only predicting
            # Predicting on validation set
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
                x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
                y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                z_val = model(x_val)
                val_pred = torch.sigmoid(z_val)
                val_preds[
                    j * val_loader.batch_size : j * val_loader.batch_size
                    + x_val[0].shape[0]
                ] = val_pred
            val_acc = accuracy_score(
                train_df.iloc[val_idx]["target"].values, torch.round(val_preds.cpu())
            )
            val_roc = roc_auc_score(
                train_df.iloc[val_idx]["target"].values, val_preds.cpu()
            )

            print(
                "Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}".format(
                    epoch + 1,
                    epoch_loss,
                    train_acc,
                    val_acc,
                    val_roc,
                    str(datetime.timedelta(seconds=time.time() - start_time))[:7],
                )
            )

            scheduler.step(val_roc)

            if val_roc >= best_val:
                best_val = val_roc
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                torch.save(model, model_path)  # Saving current best model
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping. Best Val roc_auc: {:.3f}".format(best_val))
                    break

    model = torch.load(model_path)  # Loading best model of this fold
    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        # Predicting on validation set once again to obtain data for OOF
        for j, (x_val, y_val) in enumerate(val_loader):
            x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
            x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
            y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
            z_val = model(x_val)
            val_pred = torch.sigmoid(z_val)
            val_preds[
                j * val_loader.batch_size : j * val_loader.batch_size
                + x_val[0].shape[0]
            ] = val_pred
        oof[val_idx] = val_preds.cpu().numpy()

        # Predicting on test set
        for _ in range(TTA):
            for i, x_test in enumerate(test_loader):
                x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32)
                z_test = model(x_test)
                z_test = torch.sigmoid(z_test)
                preds[
                    i * test_loader.batch_size : i * test_loader.batch_size
                    + x_test[0].shape[0]
                ] += z_test
        preds /= TTA

    del train, val, train_loader, val_loader, x, y, x_val, y_val
    gc.collect()

preds /= skf.n_splits

print("OOF: {:.3f}".format(roc_auc_score(train_df["target"], oof)))
sns.kdeplot(pd.Series(preds.cpu().numpy().reshape(-1,)))
# Saving OOF predictions so stacking would be easier
pd.Series(oof.reshape(-1,)).to_csv("oof.csv", index=False)

sub = pd.read_csv("../input/jpeg-melanoma-256x256/sample_submission.csv")
sub["target"] = preds.cpu().numpy().reshape(-1,)
sub.to_csv("submission.csv", index=False)


# %%
