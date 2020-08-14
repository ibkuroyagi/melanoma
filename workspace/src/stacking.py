# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42


def feval(pred_proba, dtrain):
    y_true = dtrain.get_label().astype(int)
    roc = roc_auc_score(y_true, pred_proba)
    return "roc", roc


# %%
test = pd.read_csv("../input/jpeg-melanoma-256x256/test.csv")
train = pd.read_csv("../input/jpeg-melanoma-256x256/train.csv")
print(test.columns)
print(train.columns)


# %%
oof_list = []
sub_list = []
No_list = [0, 1, 2]
fig_size_list = [256, 384, 512, 768]
for fig_size in fig_size_list:
    for No in No_list:
        dir_path = f"exp/{fig_size}-{No}"
        oof_path = os.path.join(dir_path, f"oof_No{No}_{fig_size}.csv")
        if os.path.isfile(oof_path):
            oof_list.append(oof_path)
        sub_path = os.path.join(dir_path, f"submission_No{No}_{fig_size}.csv")
        if os.path.isfile(sub_path):
            sub_list.append(sub_path)
print(len(oof_list), len(sub_list))


# %%
oof_cols = [f"oof_{i}" for i in range(len(oof_list))]
oof_df = pd.DataFrame(np.zeros((len(train), len(oof_list))), columns=oof_cols)
pred_cols = [f"pred_{i}" for i in range(len(oof_list))]
pred_df = pd.DataFrame(np.zeros((len(test), len(sub_list))), columns=pred_cols)
for i in range(len(oof_list)):
    oof_df.iloc[:, i] = pd.read_csv(oof_list[i]).reset_index()["0"]
    pred_df.iloc[:, i] = pd.read_csv(sub_list[i]).reset_index()["target"]


X_train, X_val, y_train, y_val = train_test_split(
    oof_df.values, train["target"].values, test_size=0.2, random_state=SEED
)
param_dist = {
    "objective": "reg:squarederror",
    "max_depth": 8,
    "learning_rate": 1e-1,
    "reg_lambda": 1e-1,
    "scale_pos_weight": 100.0,
    "random_state": SEED,
    "num_parallel_tree": 16,
    "eta": 0.1,
    "max_delta_step": 2.0,
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(pred_df.values)
evals = [(dtrain, "train"), (dval, "val")]
evals_result = {}

# kaggle competitions submit -c siim-isic-melanoma-classification -f exp/ensemble/submission_xgblogit.csv -m "Message"
param_dist = {
    "objective": "reg:linear",
    "max_depth": 8,
    "learning_rate": 1e-1,
    "reg_lambda": 10.0,
    "scale_pos_weight": 10.0,
    "lambda	": 10.0,
    "random_state": SEED,
    "num_parallel_tree": 16,
    "eta": 0.1,
    "max_delta_step": 10.0,
}
skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
sub = pd.read_csv(f"../input/jpeg-melanoma-{fig_size}x{fig_size}/sample_submission.csv")

# importance_df = pd.DataFrame(np.zeros((5, len(oof_list))), columns=oof_cols)
for i, (train_idx, val_idx) in enumerate(
    skf.split(oof_df.values, train["target"].values)
):
    X_train = oof_df.loc[train_idx].values
    X_val = oof_df.loc[val_idx].values
    y_train = train.loc[train_idx, "target"].values
    y_val = train.loc[val_idx, "target"].values

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, "train"), (dval, "val")]
    evals_result = {}
    bst = xgb.train(
        param_dist,
        dtrain,
        num_boost_round=50,
        early_stopping_rounds=15,
        evals=evals,
        evals_result=evals_result,
        feval=feval,
    )

    # importance_df.iloc[i] = bst.feature_importances_
    y_pred_proba = bst.predict(dtest)
    bst.save_model(f"exp/ensemble/xgb_{i}.h5")
    # bst.save_config("exp/ensemble/conf_xgb.json")
    sub["target"] += y_pred_proba / 5.0
sub.to_csv("exp/ensemble/submission_xgb.csv", index=False)

