from cspProblem import Constraint, CSP, Variable
from display import Displayable
import math
import pandas as pd
import matplotlib.pyplot as plt


class VincoloSoftGestioneAcqua(Constraint):
    """Vincolo soft personalizzato per gestire la crisi idrica e la distribuzione"""

    def __init__(self, scope, function, string=None, position=None):
        super().__init__(scope, function, string, position)
    
    def value(self, assignment):
        """Sovrascriviamo la funzione value per tenere conto delle penalità in caso di crisi idrica"""
        return self.holds(assignment)   #Restituisce il valore della penalità soft per l'assegnazione corrente

# Funzione per calcolare la penalità in caso di risorse insufficienti
def penalty_if_resources_insufficient(uso_acqua, risorse_disponibili, coeff_superamento=20):
    """Penalità se l'uso dell'acqua supera le risorse disponibili per una nazione"""
    if uso_acqua > risorse_disponibili:
        return (uso_acqua - risorse_disponibili) * coeff_superamento  # Più penalità per il superamento delle risorse
    return 0

def penalty_if_below_min_usage(uso_acqua,MIN_USAGE = 5):
    """Penalità se l'uso dell'acqua è inferiore al minimo desiderato"""
    if uso_acqua < MIN_USAGE:
        return (MIN_USAGE - uso_acqua) * 5
    return 0


def crea_csp_gestione_crisi_acqua(Nazioni):
    usage_vars = {nazione: Variable(f'usage_{nazione}', {i for i in range(1, 11)}) for nazione in Nazioni}
    resource_vars = {nazione: Variable(f'resources_{nazione}', {i for i in range(1, 11)}) for nazione in Nazioni}

    # Creazione dei vincoli soft per ogni nazione
    vincoli_uso_acqua = []
    for nazione in Nazioni:
        uso_var = usage_vars[nazione]
        risorsa_var = resource_vars[nazione]
        
        # Vincolo per l'uso dell'acqua che supera le risorse disponibili
        vincolo_supera_risorse = VincoloSoftGestioneAcqua(
            [uso_var, risorsa_var],
            penalty_if_resources_insufficient,
            f"Uso acqua supera le risorse disponibili per {nazione}"
        )
        vincoli_uso_acqua.append(vincolo_supera_risorse)

        # Vincolo per l'uso minimo di acqua
        vincolo_sotto_minimo = VincoloSoftGestioneAcqua(
            [uso_var],
            penalty_if_below_min_usage,
            f"Uso acqua sotto il minimo per {nazione}"
        )
        vincoli_uso_acqua.append(vincolo_sotto_minimo)

    # Creazione del CSP
    water_crisis_csp = CSP("GestioneCrisiAcquaConPenalità", 
                           list(usage_vars.values()) + list(resource_vars.values()), 
                           vincoli_uso_acqua)
    
    return water_crisis_csp


# Funzione che crea le variabili e i vincoli soft per le nazioni passate come parametro
def crea_csp_gestione_acqua(nazioni):
    """
    Crea un CSP per la gestione delle risorse idriche per una lista di nazioni.
    
    Args:
        nazioni (list): Lista di nomi delle nazioni (ad esempio, ['Italia', 'Spagna', 'Francia']).
    
    Returns:
        CSP: Un oggetto CSP configurato con le variabili e i vincoli per le nazioni.
    """
    # Step 1: Creiamo le variabili per ogni nazione (uso e risorse)
    usage_vars = {nazione: Variable(f'usage_{nazione}', {i for i in range(1, 11)}) for nazione in nazioni}
    resource_vars = {nazione: Variable(f'resources_{nazione}', {i for i in range(1, 11)}) for nazione in nazioni}

    # Step 2: Definiamo i vincoli soft
    def penalty_if_exceed_demand(uso_acqua, risorse_disponibili):
        """Vincolo soft: penalità se l'uso supera le risorse disponibili"""
        if uso_acqua > risorse_disponibili:
            return (uso_acqua - risorse_disponibili) * 2  # Penalità per superamento
        return 0  # Nessuna penalità

    def penalty_if_below_min_usage(uso_acqua,MIN_USAGE = 5):
        """Vincolo soft: penalità se l'uso è inferiore alla soglia minima"""
        # Soglia minima di uso
        if uso_acqua < MIN_USAGE:
            return (MIN_USAGE - uso_acqua) * 5  # Penalità per sottouso
        return 0

    # Step 3: Creiamo i vincoli soft per ogni nazione
    soft_constraints = []
    
    for nazione in nazioni:
        uso_var = usage_vars[nazione]
        risorsa_var = resource_vars[nazione]
        
        # Vincolo soft per l'uso che supera le risorse
        c1 = VincoloSoftGestioneAcqua([uso_var, risorsa_var], penalty_if_exceed_demand, f'uso_supera_risorse_{nazione}')
        soft_constraints.append(c1)
        
        # Vincolo soft per l'uso che è sotto la soglia minima
        c2 = VincoloSoftGestioneAcqua([uso_var], penalty_if_below_min_usage, f'uso_sotto_min_{nazione}')
        soft_constraints.append(c2)

    # Step 4: Creiamo l'istanza del CSP con tutte le variabili e i vincoli definiti
    csp = CSP("GestioneCrisiAcqua", list(usage_vars.values()) + list(resource_vars.values()), soft_constraints)

    return csp


class DF_branch_and_bound_opt(Displayable):
    """Ricercatore branch and bound per risolvere il CSP della crisi idrica"""
    def __init__(self, csp, bound=math.inf):
        """Crea un ricercatore per trovare la soluzione ottimale entro un dato limite"""
        self.csp = csp
        self.best_asst = None
        self.bound = bound

    def optimize(self):
        """Trova l'assegnazione ottimale con un costo inferiore al limite"""
        self.num_expanded = 0
        self.cbsearch({}, 0, self.csp.constraints)
        self.display(1, "Numero di percorsi espansi:", self.num_expanded)
        return self.best_asst, self.bound

    def cbsearch(self, asst, cost, constraints):
        """Trova la soluzione ottimale che estende il percorso ed è entro il limite"""
        self.display(2, "cbsearch:", asst, cost, constraints)
        can_eval = [c for c in constraints if c.can_evaluate(asst)]
        rem_cons = [c for c in constraints if c not in can_eval]
        newcost = cost + sum(c.value(asst) for c in can_eval)
        self.display(2, "Valutazione:", can_eval, "costo:", newcost)
        
        if newcost < self.bound:
            self.num_expanded += 1
            if rem_cons == []:
                self.best_asst = asst
                self.bound = newcost
                self.display(1, "Nuova migliore assegnazione:", asst, " costo:", newcost)
            else:
                var = next(var for var in self.csp.variables if var not in asst)
                for val in var.domain:
                    self.cbsearch({var: val} | asst, newcost, rem_cons)
                    

def plot_water_usage(solution):
    """
    Crea un grafico a barre per visualizzare l'uso dell'acqua per ogni nazione.
    
    Args:
        solution (dict): La soluzione CSP con allocazione dell'acqua.
    """
    nations = [var.name for var in solution.keys()]
    usage = [value for value in solution.values()]
    
    plt.figure(figsize=(20, 10))
    plt.bar(nations, usage, color='skyblue')
    plt.xlabel("Nazioni")
    plt.ylabel("Uso dell'acqua (unità)")
    plt.title("Distribuzione dell'acqua per nazione")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
