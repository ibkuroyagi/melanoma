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
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from torch.utils.data import DataLoader
import torchtoolbox.transform as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

from datasets.augmentation import AdvancedHairAugmentation, DrawHair, Microscope
from datasets.mela_datasets import MelanomaDataset
from models.net import Net
from trainers.cnn_trainer import CNNTrainer
from utils.seed import seed_everything


# %%
def get_arguments():
    """Get arguments."""
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument("--No", type=int, default=0, help="hte number of experiments.")
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save model."
    )
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
    fig_size = config["fig_size"]
    # save config
    os.makedirs(args.outdir, exist_ok=True)
    with codecs.open(f"{args.outdir}/config{No}.yaml", "w", encoding="utf-8") as f:
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
    arch = EfficientNet.from_pretrained(config["pretrained"])

    train_df, test_df, meta_features = init_df(
        train_path=f"../input/jpeg-melanoma-{fig_size}x{fig_size}/train.csv",
        test_path=f"../input/jpeg-melanoma-{fig_size}x{fig_size}/test.csv",
    )

    # setup dataloader
    skf = KFold(n_splits=config["n_split"], shuffle=True, random_state=config["seed"])
    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15)), 1):
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
        )
        val_loader = DataLoader(
            dataset=val,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )

        model = Net(
            arch=arch,
            n_meta_features=len(meta_features),
            in_features=config["in_features"],
        )  # New model for each fold
        model = model.to(device)
        logging.info(model)
        # setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            patience=config["patience"],
            verbose=True,
            factor=config["factor"],
        )

        # setup trainer
        trainer = CNNTrainer(
            steps=0,
            epochs=0,
            train_data_loader=train_loader,
            valid_data_loader=val_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
        )

        # resume from checkpoint
        if args.resume is not None:
            trainer.load_checkpoint(args.resume)
            logging.info(f"Successfully resumed from {args.resume}.")

        # start training
        try:
            trainer.run()
        except KeyboardInterrupt:
            trainer.save_checkpoint(
                os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pt")
            )
            logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")

        del train, val, train_loader, val_loader
        gc.collect()


if __name__ == "__main__":
    main()
