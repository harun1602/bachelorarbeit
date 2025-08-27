import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pyod.models.iforest import IForest
import shap
import matplotlib.pyplot as plt

## Ignorieren stand jetzt

contamination = 0.13196
n_estimators = 346
max_samples  = 'auto'
scaling      = 'standard'
train_size   = 47500
test_size    = 2497

with h5py.File("datasets/features_S01_combined_labeled.h5", "r") as f:
    # Feature‑Namen
    raw_names = f["features/S01/1/feature_names"][:]
    feature_names = [
        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else n
        for n in raw_names
    ]

    norm_grp  = f["features/S01/1/Normal/c_data"]
    relay_grp = f["features/S01/1/Relay/c_data"]
    norm_keys  = sorted(norm_grp.keys(),  key=lambda x: int(x))
    relay_keys = sorted(relay_grp.keys(), key=lambda x: int(x))

    # Training
    X_tr = np.vstack([norm_grp[k][:] for k in norm_keys[:train_size]])

    # Test
    X_te_norm  = np.vstack([norm_grp[k][:]  for k in norm_keys[-test_size:]])
    X_te_relay = np.vstack([relay_grp[k][:] for k in relay_keys[-test_size:]])
    X_te       = np.vstack((X_te_norm, X_te_relay))
    y_te       = np.array([0]*test_size + [1]*test_size)

if scaling == 'standard':
    scaler  = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_te_s  = scaler.transform(X_te)
else:
    X_tr_s, X_te_s = X_tr, X_te

model = IForest(
    contamination=contamination,
    n_estimators =n_estimators,
    max_samples  =max_samples
)
model.fit(X_tr_s)

y_pred = model.predict(X_te_s)
acc    = accuracy_score(y_te, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# SHAP‑Werte berechnen
bg          = shap.sample(X_tr_s, 200, random_state=0)
explainer   = shap.KernelExplainer(model.decision_function, bg)
X_eval      = shap.sample(X_te_s, 300, random_state=1)
shap_values = explainer.shap_values(X_eval, nsamples=200)

# Top‑50 Features nach mittlerem absoluten SHAP‑Wert
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top50_idx     = np.argsort(mean_abs_shap)[::-1][:50]

print("\nTop 50 Features nach mittlerem |SHAP|:")
for rank, idx in enumerate(top50_idx, 1):
    print(f"{rank:2d}. {feature_names[idx]:<50s} {mean_abs_shap[idx]:.4f}")

# Beeswarm plot
shap.summary_plot(
    shap_values,
    X_eval,
    feature_names=feature_names,
    show=False
)
fig_sw = plt.gcf()
fig_sw.set_size_inches(16, 10)
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png", dpi=150)
plt.clf()

# Bar plot
shap.summary_plot(
    shap_values,
    X_eval,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
fig_bar = plt.gcf()
fig_bar.set_size_inches(16, 10)
plt.tight_layout()
plt.savefig("shap_summary_bar.png", dpi=150)
plt.clf()
