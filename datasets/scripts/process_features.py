import h5py
import numpy as np

INPUT_FILE = "./raw_data/features_S01_all.h5"
OUTPUT_FILE = "./processed_datq/features_processed.npz"

def collect_data(h5file, category):
    data = []
    # Gruppen 1-10
    for i in range(1, 11):
        group_path = f"/features/S01/{i}/{category}/c_data"
        if group_path in h5file:
            for key in h5file[group_path]:
                arr = np.array(h5file[f"{group_path}/{key}"])
                data.append(arr)
    return data

with h5py.File(INPUT_FILE, "r") as f:
    normal_data = collect_data(f, "Normal")
    relay_data = collect_data(f, "Relay")

# alle längen analysieren
all_lengths = [len(d) for d in normal_data + relay_data]
min_len, max_len, avg_len = min(all_lengths), max(all_lengths), int(np.mean(all_lengths))

print(f"Minimale Länge: {min_len}")
print(f"Maximale Länge: {max_len}")
print(f"Durchschnittliche Länge: {avg_len}")

# benutzerinteraktion
target_len = int(input("Bis zu welcher Länge sollen Daten berücksichtigt werden? "))
pad_shorter = input("Kürzere Sequenzen auf Länge auffüllen (0-Padding)? (y/n): ").lower() == "y"
cut_longer = input("Längere Sequenzen kürzen? Werden sonst nicht berücksichtigt (y/n): ").lower() == "y"

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
                processed.append(arr) if L == target_len else None
        elif L > target_len:
            if cut_longer:
                processed.append(arr[:target_len])
            # else ignorieren
        else:
            processed.append(arr)
            # wenn nicht vereinheitlicht (kein pad/kein cut), dtype=object behalten
    return np.array(processed, dtype=object if not (pad_shorter or cut_longer) else np.float32)

normal_proc = process_dataset(normal_data, target_len, pad_shorter, cut_longer)
relay_proc = process_dataset(relay_data, target_len, pad_shorter, cut_longer)

np.savez(OUTPUT_FILE, normal=normal_proc, relay=relay_proc)

print(f"Daten gespeichert in {OUTPUT_FILE}")