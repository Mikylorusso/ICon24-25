import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from apprendimento import trainWaterModelKFold, oversampling, regola_gomito_e_cluster, visualizeAspectRatioChart
from prolog import carica_risorse_idriche
from csp import crea_csp_gestione_crisi_acqua, plot_water_usage, DF_branch_and_bound_opt
from pyswip import Prolog
from ragionamentoprob import bayesian_network, carica_rete_bayesiana, predizione


def main():


    # Preparazione e Normalizzazione del Dataset
    
    # Caricamento del dataset relativo alle risorse idriche.
    scaler = MinMaxScaler()
    file_csv = os.path.join(os.path.dirname(__file__), "risorse_idriche.csv")
    dataset_idrico = pd.read_csv(file_csv, sep =';',encoding='utf-8')
    print("Dataset idrico caricato. Dimensioni:", dataset_idrico.shape)
    print("Colonne:", dataset_idrico.columns.tolist())
    dataset_idrico['AcquaProCapite'] = dataset_idrico['AcquaProCapite'].astype(int)
    mediana = dataset_idrico["AcquaProCapite"].median()
    dataset_idrico["Stato_idrico"] = dataset_idrico["AcquaProCapite"].apply(lambda x: 1 if x < mediana else 0)

    nazioni = []
    if "Nazione" in dataset_idrico.columns:
        # Prendi in considerazione solo i valori unici per evitare duplicati
        nazioni = dataset_idrico["Nazione"].dropna().unique().tolist()
        print("Nazioni estratte:", nazioni)

    # Creazione di una copia per il preprocessing dedicato all'apprendimento supervisionato.
    # Vengono rimosse colonne testuali non direttamente utili ai fini del modello.
    dataset_supervised = dataset_idrico.copy()
    if "Nazione" in dataset_supervised.columns:
        dataset_supervised = dataset_supervised.drop(columns=["Nazione"])
    

    # Rimozione dei valori mancanti
    dataset_supervised.dropna(inplace=True)

    feature_cols = dataset_supervised.columns.difference(['Stato_idrico'])
    for col in feature_cols:
        dataset_supervised[col] = dataset_supervised[col].astype(str).str.replace('.','',regex=False)
        dataset_supervised[col] = pd.to_numeric(dataset_supervised[col], errors='raise').astype(int)
    dataset_supervised[feature_cols] = scaler.fit_transform(dataset_supervised[feature_cols])
    
    

    # Integrazione Prolog per la Generazione di Fatti

    # Utilizziamo un modulo dedicato per creare fatti a partire dal dataset,
    # ad esempio per generare fatti Prolog
    print("Creazione dei fatti Prolog per la gestione idrica...")
    prolog_instance = carica_risorse_idriche(file_csv, "risorse_idriche.pl")

    

    # Apprendimento Non Supervisionato

    k_elbow = regola_gomito_e_cluster(dataset_supervised)
    dataset_resampled = oversampling(dataset_supervised, "Stato_idrico")
    k_elbow = regola_gomito_e_cluster(dataset_resampled)

    

    # Apprendimento Supervisionato
    
    
    # Addestramento e validazione del modello supervisionato (ad es. per la predizione della carenza idrica).
    # Si utilizza la tecnica del KFold cross validation per garantire una valutazione robusta.
    print("Addestramento del modello supervisionato per la previsione della carenza idrica...")


    trained_models, evaluate_results = trainWaterModelKFold(dataset_supervised, target_column="Stato_idrico", resample=False)
    trained_models, evaluate_results = trainWaterModelKFold(dataset_supervised, target_column="Stato_idrico", resample=True)
    


    # Problema di Ricerca e CSP
    
    # Risoluzione del problema di allocazione delle risorse idriche.
    # Questo modulo implementa tecniche di ricerca per risolvere un CSP che rispetta i vincoli

    del dataset_supervised['Cluster']
    print("Risoluzione del problema di crisi idrica mediante tecniche CSP...")
    crisi_idrica = crea_csp_gestione_crisi_acqua(nazioni)

    bound_values = [40, 50, 60, 70, 80, 90, 100]
    num_expanded_values = []
    times = []

    for b in bound_values:
        ottimizzatore = DF_branch_and_bound_opt(crisi_idrica, bound=b)
        start_time = time.time()
        soluzione, costo = ottimizzatore.optimize()
        # Memorizziamo il numero di nodi espansi e il costo finale
        elapsed_time = time.time() - start_time
        num_expanded_values.append(ottimizzatore.num_expanded)
        times.append(elapsed_time)
        print(f"Bound: {b} - Nodi Espansi: {ottimizzatore.num_expanded}, Tempo: {elapsed_time:.4f} s, Costo: {costo}")

    max_nodes = max(num_expanded_values)
    efficiency_indices = [100 * (1 - (nodes / max_nodes)) for nodes in num_expanded_values]

    plt.figure(figsize=(8, 5))
    plt.plot(bound_values, num_expanded_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Bound Iniziale")
    plt.ylabel("Numero di Nodi Espansi")
    plt.title("Effetto del Bound sullo Spazio di Ricerca")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(bound_values, times, marker='o', linestyle='-', color='g')
    plt.xlabel("Bound Iniziale")
    plt.ylabel("Tempo di Esecuzione (s)")
    plt.title("Effetto del Bound sui Tempi di Esecuzione")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(bound_values, efficiency_indices, marker='o', linestyle='-', color='r')
    plt.xlabel("Bound Iniziale")
    plt.ylabel("Indice di Efficienza (%)")
    plt.title("Efficienza della Potatura in Funzione del Bound")
    plt.grid(True)
    plt.show()

    
    if soluzione:
        print("Soluzione CSP trovato: ")
        for var, val in soluzione.items():
            print(f"{var.name}: {val}")
        plot_water_usage(soluzione)
    else:
        print("Nessuna soluzione trovata per il CSP")

    
    # Ragionamento Probabilistico con Rete Bayesiana

    # Per lavorare con la rete bayesiana, discretizziamo le feature continue.
    discretizer = KBinsDiscretizer(n_bins=12, encode='ordinal', strategy='uniform')
    dataset_disc = dataset_supervised.copy()
    continuous_cols = dataset_disc.select_dtypes(include=['float64', 'int64']).columns
    dataset_disc[continuous_cols] = discretizer.fit_transform(dataset_disc[continuous_cols])
    print("Dati discretizzati per l'analisi della rete bayesiana.")
    
    # Carichiamo la rete bayesiana esistente oppure ne creiamo una nuova se necessario.
    #try:
    #water_bn = carica_rete_bayesiana()
    #print("Rete bayesiana caricata con successo.")
    #except Exception as e:
    print("Impossibile caricare una rete bayesiana esistente. Creazione di una nuova rete...")
    water_bn = bayesian_network(dataset_disc)
    
    # Esempio pratico: utilizzo della rete bayesiana per predire la presenza di carenza idrica.
    print("Esecuzione predizione con la rete bayesiana per un caso di studio...")
    example_event = dataset_disc.iloc[0].to_dict()  # Utilizziamo la prima osservazione come esempio

    print("Evento idrico generato per predizione:", example_event)
    predizione(water_bn, example_event, "Stato_idrico")
