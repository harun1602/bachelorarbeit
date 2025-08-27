import h5py
import numpy as np
from collections import Counter

INPUT_FILE = "./raw_data/transients_S01_all.h5"
OUTPUT_FILE = "./processed_data/transients_processed.npz"

def collect_data(h5file, category):
    data = []
    # Gruppen 1-10
    for i in range(1, 11):  # Gruppen 1-10
        group_path = f"/transients/S01/{i}/{category}/q_data"
        if group_path in h5file:
            for key in h5file[group_path]:
                arr = np.array(h5file[f"{group_path}/{key}"])
                data.append(arr)
    return data

def most_common_len(lengths):
    if not lengths:
        return None, 0, 0.0
    c = Counter(lengths).most_common(1)[0]
    mode_len, count = c[0], c[1]
    return mode_len, count, count / len(lengths)

with h5py.File(INPUT_FILE, "r") as f:
    normal_data = collect_data(f, "Normal")
    relay_data  = collect_data(f, "Relay")

# längen analysieren
normal_lengths = [len(d) for d in normal_data]
relay_lengths  = [len(d) for d in relay_data]
all_lengths    = normal_lengths + relay_lengths

min_len  = min(all_lengths)
max_len  = max(all_lengths)
avg_len  = int(np.mean(all_lengths))

# häufigste länge (gesamt + pro Kategorie)
mode_all,   count_all,   ratio_all   = most_common_len(all_lengths)
mode_norm,  count_norm,  ratio_norm  = most_common_len(normal_lengths)
mode_relay, count_relay, ratio_relay = most_common_len(relay_lengths)

print(f"Minimale Länge: {min_len}")
print(f"Maximale Länge: {max_len}")
print(f"Durchschnittliche Länge: {avg_len}")
print(f"Häufigste Länge (gesamt): {mode_all}  | Anzahl: {count_all}  | Anteil: {ratio_all:.1%}")
print(f"Häufigste Länge (Normal): {mode_norm} | Anzahl: {count_norm} | Anteil: {ratio_norm:.1%}")
print(f"Häufigste Länge (Relay):  {mode_relay} | Anzahl: {count_relay} | Anteil: {ratio_relay:.1%}")

# benutzerinteraktion
target_len   = int(input("Bis zu welcher Länge sollen Daten berücksichtigt werden? "))
pad_shorter  = input("Kürzere Sequenzen auf Länge auffüllen (0-Padding)? (y/n): ").lower() == "y"
cut_longer   = input("Längere Sequenzen kürzen? Werden sonst nicht berücksichtigt (y/n): ").lower() == "y"

def process_dataset(dataset, target_len, pad_shorter, cut_longer):
    processed = []
    for arr in dataset:
        L = len(arr)
        if L < target_len:
            if pad_shorter:
                padded = np.zeros(target_len, dtype=arr.dtype)
                padded[:L] = arr
                processed.append(padded)
            else:
                if L == target_len:
                    processed.append(arr)
        elif L > target_len:
            if cut_longer:
                processed.append(arr[:target_len])
                # else ignorieren
        else:
            processed.append(arr)
            # wenn nicht vereinheitlicht (kein pad/kein cut), dtype=object behalten
    return np.array(processed, dtype=object if not (pad_shorter or cut_longer) else np.float32)

normal_proc = process_dataset(normal_data, target_len, pad_shorter, cut_longer)
relay_proc  = process_dataset(relay_data,  target_len, pad_shorter, cut_longer)

np.savez(OUTPUT_FILE, normal=normal_proc, relay=relay_proc)

print(f"Daten gespeichert in {OUTPUT_FILE}")
