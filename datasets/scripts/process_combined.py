import numpy as np

INPUT_FILE1 = "./processed_data/transients_processed.npz"
INPUT_FILE2 = "./processed_data/features_processed.npz"
OUTPUT_FILE = "./processed_data/combined_processed.npz"

def load_npz(path):
    with np.load(path, allow_pickle=True) as f:
        if "normal" not in f or "relay" not in f:
            raise KeyError(f"[ERROR] {path} muss Keys 'normal' und 'relay' enthalten.")
        return f["normal"], f["relay"]

def main():
    t_norm, t_relay = load_npz(INPUT_FILE1)
    f_norm, f_relay = load_npz(INPUT_FILE2)

    # keine unregelmäßigen arrays untereinander prüfen
    for name, arr in [("transients normal", t_norm), ("transients relay", t_relay),
                      ("features normal", f_norm), ("features relay", f_relay)]:
        if arr.dtype == object:
            raise ValueError(f"[ERROR] {name} enthält unregelmäßige Längen.")

    # gleiche anzahl samples prüfen
    if not (t_norm.shape[0] == f_norm.shape[0] and t_relay.shape[0] == f_relay.shape[0]):
        raise ValueError("[ERROR] Unterschiedliche Anzahl Samples zwischen Transients und Features.")

    # kombinieren
    combined_normal = np.concatenate([t_norm, f_norm], axis=1)
    combined_relay  = np.concatenate([t_relay, f_relay], axis=1)

    print(f"Transients normal={t_norm.shape}, relay={t_relay.shape}")
    print(f"Features   normal={f_norm.shape}, relay={f_relay.shape}")
    print(f"Combined   normal={combined_normal.shape}, relay={combined_relay.shape}")

    np.savez(OUTPUT_FILE, normal=combined_normal.astype(np.float32, copy=False),
             relay=combined_relay.astype(np.float32, copy=False))
    print(f"Gespeichert: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
