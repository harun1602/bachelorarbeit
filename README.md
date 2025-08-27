## Nutzung der Process Skripte
Sicher gehen, dass die unbearbeiteten Signale 

Process Dateien in folgernder Reihenfolge öffnen:
1. process_transients.py
2. process_features.py
3. process_combined.py
4. process_fouriers.py

Mögliche Änderungen des Codes:

INPUT_FILE und OUTPUT_FILE Variablen in Zeile 5 und 6

group_path in Zeile 12. In den Feature Daten werden die Daten zum Beispiel
unter einem c_data Pfad gespeichert, während es in den transients Daten
unter q_data der Fall ist. Hier einfach ändern.

## Nutzung der main.py
Argumente: 

--modell        required = True, {knn, lof, gmm, if, autoencoder, deepsvdd, lunar}  
--projectname   required = True, {Project name of choice}  
--count         required = True, {Number of runs per sweep}  
--n_train       required = True, {(Max) Training Data}  
--enable_n_train_split if set, Training Data will be split into Data into Blocks of X\*n_trian_step up to n_train  
--n_train_step  required = True if --enable_n_train_split is set, default = 1000, {1000, 0.95\*n_train}  
--sweep_method  required = False, default = bayes, {bayes, random, grid}  
--metric_name   required = False, default = accuracy, {accuracy, f1, youden_j}  
--metric_goal   required = False, default = maximize, {maximize, minimize}  
--dataset       required = False, default = normal, features, combined, fouriers, {normal, features, combined, fouriers}  
--scaling       required = False, default = raw, {raw, minmax, standard} 
