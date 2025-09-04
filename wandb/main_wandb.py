import argparse
import math
import numpy as np
import wandb
import matplotlib.pyplot as plt
import time

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.gmm import GMM
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lunar import LUNAR

datasets_info = {
    'normal': {
        'file_path':    "../datasets/processed_data/transients_processed.npz"
    },
    'features': {
        'file_path':    "../datasets/processed_data/features_processed.npz"
    },
    'combined': {
        'file_path':    "../datasets/processed_data/combined_processed.npz"
    },
    'fouriers': {
        'file_path':    "../datasets/processed_data/fouriers_processed.npz"
    }
}

ALLOWED_METHODS = {"bayes", "random", "grid"}
ALLOWED_METRICS = {"accuracy", "f1", "auc"}
ALLOWED_SCALINGS = {"raw", "minmax", "standard"}
ALLOWED_DATASETS = {"normal", "features", "combined", "fouriers"}

# basis-Sweeps (werden zur laufzeit überschrieben/angepasst)
SWEEPS = {
    "knn":
        {
        'method':            {},
        'metric':            {},
        'parameters':
            {
            'radius':        {'min': 10, 'max': 50},
            'method':        {'values': ['largest', 'median']},
            'n_neighbors':   {'min': 1, 'max': 300},
            'contamination': {'min': 0.01, 'max': 0.15},
            'leaf_size':     {'min': 1, 'max': 100},
            'algorithm':     {'values': ['auto']},
            'metric':        {'values': ["euclidean", "minkowski"]},
            'dataset':       {'values': ['normal', 'features', 'combined']},
            'scaling':       {'values': []},
            'n_train':       {'values': []},
            }
        },
    "lof":
        {
        'method':            {},
        'metric':            {},
        'parameters':
            {
            'n_neighbors':   {'min': 1,    'max': 800},
            'contamination': {'min': 0.01, 'max': 0.15},
            'leaf_size':     {'min': 1,    'max': 200},
            'algorithm':     {'values': ['auto']},
            'novelty':       {'values': [True]},
            'metric':        {'values': ["euclidean", "manhattan", "chebyshev", "minkowski"]},
            'dataset':       {'values': ['normal', 'features', 'combined']},
            'scaling':       {'values': []},
            'n_train':       {'values': []},
            }
        },
    "gmm":
        {
        'method':              {},
        'metric':              {},
        'parameters':
            {
            'n_components':    {'min': 1,     'max': 100},
            'contamination':   {'min': 0.01,  'max': 0.15},
            'reg_covar':       {'min': 1e-06, 'max': 1e-04},
            'max_iter':        {'value': 1000},
            'tol':             {'min': 1e-04, 'max': 1e-02},
            'covariance_type': {'values': ['full', 'tied', 'diag', 'spherical']},
            'dataset':         {'values': ['normal', 'features', 'combined']},
            'scaling':         {'values': []},
            'n_train':         {'values': []},
            }
        },
    "if":
        {
        'method':            {},
        'metric':            {},
        'parameters':
            {
            'contamination': {'min': 0.01, 'max': 0.15},
            'n_estimators':  {'min': 200, 'max': 1000},
            'behaviour':     {'values': ['new', 'old']},
            'dataset':       {'value': 'features'},
            'scaling':       {'values': []},
            'n_train':       {'values': []},
            }
        },
    "autoencoder":
        {
        'method':              {},
        'metric':              {},
        'parameters':
            {
            'contamination':          {'min': 0.01,  'max': 0.15},
            'preprocessing':          {'values': [True, False]},
            'lr':                     {'distribution': 'log_uniform', 'min': 1e-08, 'max': 1e-02},
            'epoch_num':              {'values': [20, 100]},
            'batch_size':             {'distribution': 'int_uniform', 'min': 32, 'max': 128},
            'hidden_neuron_list':     {'values': [[64, 32], [128, 64], [256, 128], [256, 64], [128, 64, 32], [256, 128, 64, 32]]},
            'hidden_activation_name': {'values': ['relu', 'tanh', 'sigmoid']},
            'dropout_rate':           {'distribution': 'uniform', 'min': 0.0001, 'max': 1.0},
            'dataset':                {'values': ['normal', 'features', 'combined']},
            'scaling':                {'values': []},
            'n_train':                {'values': []},
            }
        },
    "lunar":
        {
        'method':                {},
        'metric':                {},
        'parameters':
            {
            'n_neighbours':      {'min': 1,    'max': 1000},
            'model_type':        {'values': ["WEIGHT"]},
            'negative_sampling': {'values': ['UNIFORM', 'SUBSPACE', 'MIXED']},
            'val_size':          {'min': 0.01, 'max': 0.8},
            'contamination':     {'min': 0.01, 'max': 0.15},
            'epsilon':           {'min': 0.01, 'max': 1.0},
            'proportion':        {'min': 1.0,  'max': 50.0},
            'n_epochs':          {'min': 100, 'max': 500},
            'lr':                {'min': 0.001,'max': 0.01},
            'wd':                {'min': 0.2, 'max': 1.0},
            'dataset':           {'values': ['normal', 'features', 'combined']},
            'scaling':           {'values': []},
            'n_train':           {'values': []},
            }
        }
}

# returned N = anzahl samples im normal-set (relay hat gleiche länge)
def get_dataset_length(name) -> int:
    info = datasets_info.get(name)
    if info is None:
        raise ValueError(f"Unbekanntes Dataset: {name}")
    with np.load(info['file_path'], allow_pickle=True) as f:
        N = f["normal"].shape[0]
    return N

# baut trainings und testdaten für den aktuellen run
# train: erste current_n_train samples aus normal
# test:  letzte floor(0.05 * current_n_train) samples aus "ende" von normal und relay (konkateniert)
def load_dataset(name, current_n_train, scaling):
    info = datasets_info.get(name)
    if info is None:
        raise ValueError(f"Unbekanntes Dataset: {name}")

    with np.load(info['file_path'], allow_pickle=True) as f:
        data_normal = f["normal"]
        data_relay = f["relay"]

    X_train = data_normal[:current_n_train]
    n_test_each = int(math.floor(0.05 * current_n_train))
    test_normal = data_normal[-n_test_each:]
    test_relay = data_relay[-n_test_each:]

    X_test = np.concatenate([test_normal, test_relay], axis=0)
    y_test = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_relay))], axis=0)

    if scaling == 'minmax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaling == 'standard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_test

def evaluate_and_log(clf, model_name, y_test, X_test):
    # startet laufzeit
    start_time = time.time()

    y_pred = clf.predict(X_test)
    y_scores = clf.decision_function(X_test)

    # berechnet vergangene zeit
    runtime = time.time() - start_time

    run_name = wandb.run.name

    # accuracy score, classification report
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    clf_report = classification_report(y_test, y_pred, target_names=["Normal", "Relay"], zero_division=0)

    # ROC und AUC
    fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=1)
    auc = roc_auc_score(y_test, y_scores)
    fig_roc, ax_roc = plt.subplots(figsize=(6,6), dpi=200)
    ax_roc.plot(fpr, tpr, lw=2)
    ax_roc.plot([0, 1], [0, 1], '--', lw=1)
    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=f"ROC\nModell: {model_name}\nRun: {run_name}")
    roc = wandb.Image(fig_roc)
    plt.close(fig_roc)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(dpi=200)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Relay"])
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues)
    ax_cm.set_title(f"Confusion Matrix\nModell: {model_name}\nRun: {run_name}")
    fig_cm.tight_layout()
    plt.close(fig_cm)

    wandb.log({
        "accuracy": acc,
        "f1": f1,
        "classification_report": wandb.Html(f"<pre>{clf_report}</pre>"),
        "confusion_matrix": wandb.Image(fig_cm),
        "pred_runtime": runtime,
        "roc": wandb.Image(roc),
        "auc": auc
    })

# trainingsfunktion für sweep
def train(config=None):
    with wandb.init(config=config):
        cfg = wandb.config
        modell = cfg.get("modell")
        dataset = cfg.get("dataset")
        scaling = cfg.get("scaling")
        n_train = int(cfg.get("n_train"))
        contamination = float(cfg.get("contamination"))

        X_train, X_test, y_test = load_dataset(dataset, n_train, scaling)

        if modell == "knn":
            clf = KNN(
                contamination=contamination,
                n_neighbors=int(cfg.get("n_neighbors", 5)),
                algorithm=cfg.get("algorithm", "auto"),
                leaf_size=int(cfg.get("leaf_size", 30)),
                metric=cfg.get("metric", "euclidean"),
                method=cfg.get("method", "largest"),
                radius=int(cfg.get("radius", 20))
            )
            clf.fit(X_train)
        elif modell == "lof":
            clf = LOF(
                contamination=contamination,
                n_neighbors=int(cfg.get("n_neighbors", 20)),
                leaf_size=int(cfg.get("leaf_size", 30)),
                algorithm=cfg.get("algorithm", "auto"),
                novelty=bool(cfg.get("novelty", True)),
                metric=cfg.get("metric", "euclidean")
            )
            clf.fit(X_train)
        elif modell == "if":
            clf = IForest(
                contamination=contamination,
                n_estimators=int(cfg.get("n_estimators", 100)),
                behaviour=cfg.get("behaviour", "new")
            )
            clf.fit(X_train)
        elif modell == "gmm":
            clf = GMM(
                n_components=int(cfg.get("n_components", 1)),
                covariance_type=cfg.get("covariance_type", "full"),
                max_iter=int(cfg.get("max_iter", 1000)),
                tol=float(cfg.get("tol", 1e-3)),
                reg_covar=float(cfg.get("reg_covar", 1e-6))
            )
            clf.fit(X_train)
        elif modell == "autoencoder":
            clf = AutoEncoder(
                hidden_neuron_list=cfg.get("hidden_neuron_list"),
                hidden_activation_name=cfg.get("hidden_activation_name"),
                dropout_rate=cfg.get("dropout_rate"),
                lr=cfg.get("lr"),
                epoch_num=int(cfg.get("epoch_num")),
                batch_size=int(cfg.get("batch_size")),
                preprocessing=bool(cfg.get("preprocessing"))
            )
            clf.fit(X_train)
        elif modell == "lunar":
            clf = LUNAR(
                n_neighbours=int(cfg.get("n_neighbours")),
                model_type=cfg.get("model_type"),
                negative_sampling=cfg.get("negative_sampling"),
                val_size=cfg.get("val_size"),
                contamination=cfg.get("contamination"),
                epsilon=cfg.get("epsilon"),
                proportion=cfg.get("proportion"),
                n_epochs=int(cfg.get("n_epochs")),
                lr=cfg.get("lr"),
                wd=cfg.get("wd"),
                scaler=StandardScaler()
            )
            clf.fit(X_train)
        else:
            raise NotImplementedError(f"Modell '{modell}' nicht implementiert.")

        evaluate_and_log(clf, modell, y_test, X_test)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--modell",
                        type=str,
                        required=True,
                        choices=list(SWEEPS.keys()),
                        help=f"Welches Modell benutzen (knn, lof, gmm, if, autoencoder, lunar)\ndefault: {list(SWEEPS.keys())}")
    parser.add_argument(
                        "--count",
                        type=int,
                        required=True,
                        help="Wie viele Runs der Agent ausführen soll")
    parser.add_argument(
                        "--projectname",
                        type=str,
                        required=True,
                        help="WandB Projektname"
                        )
    parser.add_argument(
                        "--sweep_method",
                        type=str,
                        default="bayes",
                        choices=list(ALLOWED_METHODS),
                        help="Sweep-Optimierungsmethode (bayes, random, grid)\ndefault: bayes"
                        )
    parser.add_argument(
                        "--metric_name",
                        type=str,
                        default="accuracy",
                        choices=list(ALLOWED_METRICS),
                        help="Metrik für die Sweep-Auswertung (accuracy, f1, auc)\ndefault: accuracy"
                        )
    parser.add_argument(
                        "--metric_goal",
                        type=str,
                        default="maximize",
                        choices={"maximize", "minimize"},
                        help="Ziel der Metrik (maximize, minimize)\ndefault: maximize"
                        )
    parser.add_argument(
                        "--dataset",
                        type=str,
                        nargs="*",
                        default=None,
                        choices=list(ALLOWED_DATASETS),
                        help=f"Ein oder mehrere Datensätze (normal, features, combined, fouriers). Ohne Angabe: alle vier.\ndefault: {list(ALLOWED_DATASETS)}"
                        )
    parser.add_argument(
                        "--scaling",
                        type=str,
                        default=None,
                        choices=list(ALLOWED_SCALINGS),
                        help="Skalierungsmethode (raw, standard, minmax)\ndefault: raw"
                        )
    parser.add_argument(
                        "--n_train",
                        type=int,
                        required=True,
                        help="Ziel-Trainingsgröße."
                        )
    parser.add_argument(
                        "--enable_n_train_split",
                        action="store_true",
                        help="Wenn gesetzt, erstellt die Sweep-Konfiguration n_train-Splits"
                        )
    parser.add_argument(
                        "--n_train_step",
                        type=int,
                        help="Schrittweite der n_train-Splits (nur nötig bei --enable_n_train_split)"
                        )

    args = parser.parse_args()

    # default: alle datasets und scalings, wenn nichts angegeben wurde. spinnt sonst irgendwie
    if not args.dataset:
        args.dataset = list(ALLOWED_DATASETS)

    if not args.scaling:
        args.scaling = list(ALLOWED_SCALINGS)

    modell = args.modell

    # datensatzlängen prüfen und strengstes n_train-limit bestimmen also N*0.95
    total_N_per_dataset = {}
    for dataset in args.dataset:
        total_N = get_dataset_length(dataset)
        if total_N <= 0:
            raise ValueError(f"Dataset '{dataset}' scheint leer zu sein (N={total_N}).")
        total_N_per_dataset[dataset] = total_N

    max_allowed_train_global = min(int(math.floor(0.95 * N)) for N in total_N_per_dataset.values())

    if args.n_train < 1000:
        print(f"[WARN] Angegebenes n_train={args.n_train} < 1000. Setze auf mindestens 1000.")
        user_n_train = 1000
    else:
        user_n_train = args.n_train

    if user_n_train > max_allowed_train_global:
        # alle datasets finden, die das minimale limit haben
        limiting_datasets = [ds for ds, N in total_N_per_dataset.items()
                             if int(math.floor(0.95 * N)) == max_allowed_train_global]

        limiting_str = ", ".join(limiting_datasets)

        print(f"[WARN] n_train={user_n_train} > globalem Maximum {max_allowed_train_global} "
              f"(strengstes 95%-Limit wegen Dataset(s): {limiting_str}). "
              f"Setze auf {max_allowed_train_global}.")
        user_n_train = max_allowed_train_global

    # n_train-Liste erzeugen
    if args.enable_n_train_split:
        if args.n_train_step is None:
            parser.error("--n_train_step muss angegeben werden, wenn --enable_n_train_split gesetzt ist.")
        step = int(args.n_train_step)
        values = list(range(step, user_n_train + 1, step))
        if values[-1] != user_n_train:
            values.append(user_n_train)
        n_train_values = values
    else:
        n_train_values = [user_n_train]

    # sweep-konfiguration vorbereiten/überschreiben
    sweep_conf                          = SWEEPS[modell].copy()
    sweep_conf['method']                = args.sweep_method
    sweep_conf['metric']                = {'name': args.metric_name, 'goal': args.metric_goal}
    sweep_conf['parameters']            = sweep_conf['parameters'].copy()
    sweep_conf['parameters']['dataset'] = {'values': args.dataset}
    sweep_conf['parameters']['scaling'] = {'values': args.scaling}
    sweep_conf['parameters']['n_train'] = {'values': n_train_values}
    sweep_conf['parameters']['modell']  = {'value' : args.modell}

    # sweep starten
    sweep_id = wandb.sweep(sweep_conf, project=args.projectname)
    wandb.agent(sweep_id, function=train, count=args.count)

if __name__ == "__main__":
    main()
