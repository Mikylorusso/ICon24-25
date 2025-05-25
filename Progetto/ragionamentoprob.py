from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, K2, BIC
from pgmpy.inference import VariableElimination
from pgmpy.metrics import log_likelihood_score
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def bayesian_network(dataset):
    """
    Costruzione avanzata di una rete bayesiana con apprendimento della struttura e inferenza complessa.
    """
    # Caricamento del dataset preprocessato
    
    # Apprendimento della struttura (Hill Climb Search con punteggio K2)
    hc = HillClimbSearch(dataset)
    best_model_structure = hc.estimate(scoring_method=K2(dataset))
    print("Struttura della rete appresa:")
    print(best_model_structure.edges())
    
    # Creazione del modello basato sulla struttura appresa
    model = DiscreteBayesianNetwork(best_model_structure.edges())
    
    # Apprendimento dei parametri con stime bayesiane
    model.fit(dataset, estimator=MaximumLikelihoodEstimator, n_jobs=-1)

    
    cpds = model.get_cpds()
    print("CPDs assegnati nel modello:")
    for cpd in cpds:
        print(f"\nCPD per la variabile '{cpd.variable}':")
        print(cpd)
        print("\n-----------------------------------------------------------------------------------------")

    
    # Validazione log-likelihood
    log_likelihood = log_likelihood_score(model, dataset)
    bic_score = BIC(dataset)
    bic_value = bic_score.score(model)
    print("BIC:", bic_value)

    print(f"Log-Likelihood del modello: {log_likelihood}")

     # Salvataggio della struttura e dei parametri
    try:
        with open('modello_bayesiano.pkl', 'wb') as output_file:
            pickle.dump(model, output_file)
        print(f"Modello bayesiano salvato in '{modello_bayesiano.pkl}'")
    except Exception as e:
        print("Errore durante il salvataggio del modello:", e)
    visualizza_grafo_bayesian_network(model)

    return model
    
def carica_rete_bayesiana():
    with open('modello_bayesiano.pkl','rb') as input:
       model = pickle.load(input)
    visualizza_grafo_bayesian_network(model)
    return model

def predizione(model, evidences, target_column):

    evidences = evidences.copy()
    if target_column in evidences:
        del evidences[target_column]
     # Inferenza con Variable Elimination
    inference = VariableElimination(model)
    
    # Inferenza complessa: es. probabilità congiunta di Stato_Idrico dato Acqua Annuale e Popolazione
    joint_query = inference.query(variables=[target_column], evidence=evidences)
    print("Risultati Inferenza Congiunta:")
    print(joint_query)

"""def predizione1(model, dataset, target_column):
    inference = VariableElimination(model)
    risultati = []

    for index, row in dataset.iterrows():
        evidences = row.to_dict()
        evidences.pop(target_column, None)
        query_result = inference.query(variables=[target_column], evidence=evidences)
        risultati.append(query_result)
        print(f"Osservazione {index} - predizione per {target_column}: \n", query_result)

    return risultati"""

def visualizza_grafo_bayesian_network(model):
    """
    Visualizza il grafo di una rete bayesiana.

    Parametri:
      model: Oggetto che rappresenta la rete bayesiana (ad esempio un'istanza di pgmpy.models.BayesianModel).
             Si assume che model abbia i metodi nodes() ed edges() per ottenere nodi e archi.
    """
    # Crea un grafo diretto 
    Grafo = nx.MultiDiGraph()
    
    # Aggiungi nodi e archi dal modello della rete bayesiana
    Grafo.add_nodes_from(model.nodes())
    Grafo.add_edges_from(model.edges())
    
    # Calcola un layout per posizionare i nodi in modo gradevole
    pos = nx.spring_layout(Grafo)
    
    # Disegna il grafo
    nx.draw(Grafo, pos,
            with_labels=True,
            node_size=2000,
            node_color='lightblue',
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20)
    
    plt.title("Grafo della Rete Bayesiana")
    plt.show()
    plt.clf()
