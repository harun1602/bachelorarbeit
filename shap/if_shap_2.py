import h5py
import numpy as np
import wandb
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from pyod.models.iforest import IForest

## Ignorieren stand jetzt

INPUT_PATH = "datasets/features_S01_combined_labeled.h5"
selected_features = [
    "0_Peak to peak distance","0_LPCC_0","0_Min","0_Spectral skewness",
    "1_std_abs","0_Max","0_Spectral kurtosis","0_Spectrogram mean coefficient_225.81Hz",
    "0_Spectral spread","0_LPCC_2","0_Spectrogram mean coefficient_451.61Hz",
    "0_ECDF_1","0_Variance","0_LPCC_11","0_LPCC_10",
    "0_Wavelet standard deviation_83.33Hz","0_Wavelet variance_166.67Hz"
]

with h5py.File(INPUT_PATH, "r") as f:
    raw_names = f["features/S01/1/feature_names"][:]
    all_features = [
        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else n
        for n in raw_names
    ]
    sel_idx = [all_features.index(ftr) for ftr in selected_features]

    norm_grp  = f["features/S01/1/Normal/c_data"]
    relay_grp = f["features/S01/1/Relay/c_data"]
    norm_keys  = sorted(norm_grp.keys(),  key=lambda x: int(x))
    relay_keys = sorted(relay_grp.keys(), key=lambda x: int(x))

    split_idx = 47500
    test_size = 2497

    pool_all = np.vstack([norm_grp[k][:] for k in norm_keys[:split_idx]])

    X_te_norm_all  = np.vstack([norm_grp[k][:]  for k in norm_keys[-test_size:]])
    X_te_relay_all = np.vstack([relay_grp[k][:] for k in relay_keys[-test_size:]])
    X_test_all     = np.vstack((X_te_norm_all, X_te_relay_all))
    y_test         = np.array([0]*test_size + [1]*test_size)

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'accuracy', 'goal': 'maximize'},
    'parameters': {
        'contamination': {'distribution': 'uniform', 'min': 0.05, 'max': 0.15},
        'n_estimators':  {'min': 10, 'max': 1000},
        'max_samples':   {'values': ['auto', 0.5, 1.0]},
        'scaling':       {'values': ['raw', 'minmax', 'standard']},
        'n_train':       {'values': [100, 1000, 10000, 20000, 47500]},
    }
}
sweep_id = wandb.sweep(sweep_config, project='pyod_if_bayes_shap')

def train():
    run = wandb.init()
    cfg = run.config

    # Trainingsset: erste cfg.n_train Samples aus dem Pool, auf Top‑50 reduziert
    n_req = int(cfg.n_train)
    n_avl = pool_all.shape[0]
    n     = min(n_req, n_avl)
    X_tr_all = pool_all[:n]
    X_tr     = X_tr_all[:, sel_idx]
    X_te     = X_test_all[:, sel_idx]
    y_te     = y_test

    run.log({'train_size': n})

    if cfg.scaling == 'minmax':
        scaler = MinMaxScaler().fit(X_tr)
        X_tr_s  = scaler.transform(X_tr)
        X_te_s  = scaler.transform(X_te)
    elif cfg.scaling == 'standard':
        scaler = StandardScaler().fit(X_tr)
        X_tr_s  = scaler.transform(X_tr)
        X_te_s  = scaler.transform(X_te)
    else:
        X_tr_s, X_te_s = X_tr, X_te

    model = IForest(
        contamination=cfg.contamination,
        n_estimators=int(cfg.n_estimators),
        max_samples=cfg.max_samples
    )
    model.fit(X_tr_s)



    y_pred = model.predict(X_te_s)
    acc    = accuracy_score(y_te, y_pred)

    cm     = confusion_matrix(y_te, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=["Normal", "Relay"])
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4), dpi=200)
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
    ax_cm.set_title(f"Top50 — train size {n}")
    plt.tight_layout()

    report_html = wandb.Html(
        f"<pre>{classification_report(y_te, y_pred, target_names=['Normal','Relay'])}</pre>"
    )

    scores      = model.decision_function(X_te_s)
    scores_norm = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
    preds_proba = np.vstack([1 - scores_norm, scores_norm]).T
    pr = wandb.plot.pr_curve(
        y_true=y_te,
        y_probas=preds_proba,
        labels=["Normal", "Relay"]
    )
    wandb.log({
        "confusion_matrix":      wandb.Image(fig_cm),
        "pr_curve":              pr,
        "accuracy":              acc,
        "classification_report": report_html
    })



    run.finish()

wandb.agent(sweep_id, function=train, count=1000)
