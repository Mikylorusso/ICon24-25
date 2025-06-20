import pandas as pd
import numpy as np
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score, learning_curve
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score, silhouette_samples
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
from imblearn.over_sampling import SMOTE

def trainWaterModelKFold(df, target_column = "Stato_idrico", resample=False):
    
    # Selezione delle feature e del target
    # Utilizzo delle feature 'Acqua Annuale' e 'Popolazione'
    X = df[['AcquaAnnuale', 'Popolazione']]
    y = df[target_column]

    # Definizione dello schema di cross validation (k-fold stratificato)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Suddivisione in training e test set (80/20); stratifichiamo rispetto al target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if resample:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Definizione dei modelli e dei relativi iperparametri da ottimizzare
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000),
            "param_grid": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            },
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            },
        },
    }
    
    # Dizionari per salvare le predizioni e i modelli addestrati
    trained_models = {}
    evaluation_results = {}
    
    # Per ogni modello, eseguiamo Grid Search, valutiamo in cross validation e calcoliamo le metriche
    for name, config in models.items():
        print(f"\n----- Valutazione per il modello: {name} -----")
        model = config["model"]
        param_grid = config["param_grid"]

        # Fase di Grid Search per la selezione dei migliori iperparametri
        print("Esecuzione della Grid Search per ottimizzazione degli iperparametri...")
        grid_search = GridSearchCV(estimator=model, param_grid = param_grid, scoring = "accuracy", cv=skf, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Migliori iperparametri trovati per {name}: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_

        # Valutazione tramite validazione incrociata
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring='accuracy')
        #print(f"CV Accuracy scores: {cv_scores}")
        #print(f"Media Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
       
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        # Calcolo delle metriche sul test set
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, normalize='true')

        print("\nValutazione sul test set:")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        print("Matrice di Confusione:")
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Salvataggio del modello addestrato su file
        model_filename = f"{name}_model.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(best_model, file)
        print(f"Modello {name} addestrato e salvato in: {model_filename}")

        # Salvataggio delle predizioni ottenute per ogni modello
        output_df = X_test.copy()
        output_df['Reale'] = y_test.values
        output_df['Predetto'] = y_pred
        prediction_filename = f"predictions_{name}.csv"
        output_df.to_csv(prediction_filename, index=False)
        print(f"Predizioni per {name} salvate in: {prediction_filename}")
        print("----------------------------------------------------------------------------------")

        trained_models[name] = best_model
        evaluation_results[name] = {
              "best_params":grid_search.best_params_,
              "cv_scores":cv_scores,
              "test_accuracy": accuracy,
              "test_precision": precision,
              "test_recall": recall,
              "test_f1":f1,
              "confusion_matrix": cm,
        }

    
    print("Fase di training e valutazione completata.")

    plot_learning_curves(trained_models["LogisticRegression"], X, y, target_column, 'LogisticRegression')
    plot_learning_curves(trained_models["DecisionTree"], X, y, target_column, 'DecisionTree')
    plot_learning_curves(trained_models["RandomForest"], X, y, target_column, 'RandomForest')
    plot_learning_curves(trained_models["KNN"], X, y, target_column, 'KNN')
    visualizeMetricsGraphs(evaluation_results)

    return trained_models, evaluation_results


def regola_gomito_e_cluster(dataset):

    if "Stato_idrico" in dataset.columns:
        dataset1 = dataset.drop(columns=["Stato_idrico"])
    else:
        dataset1 = dataset.copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset1)
    inertia = []
    K = range(1,11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    k1 = KneeLocator(K, inertia, curve="convex", direction="decreasing")
    plt.figure(figsize=(8,4))
    plt.plot(K,inertia, marker='o')
    plt.scatter(k1.elbow, inertia[k1.elbow - 1], c='red', label = f'Miglior k: {k1.elbow}')
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Regola del gomito per selezionare k')
    plt.legend()
    plt.show()


    kmeans_final = KMeans(n_clusters=k1.elbow, random_state=42)
    clusters = kmeans_final.fit_predict(scaled_data)
    dataset['Cluster'] = clusters
    print(dataset.head())
    sil_score = silhouette_score(scaled_data, clusters)
    print("Silhouette Score globale: {:.3f}".format(sil_score))

    sample_silhouette_values = silhouette_samples(scaled_data, clusters)

    fig, ax = plt.subplots(figsize=(8, 6))

    y_lower = 10  # Iniziamo dallo spazio 10 pixel più in basso per raggruppare i plot
    for cluster in range(k1.elbow):
        # Estraiamo i valori per il cluster i-esimo e dopo ordinati
        cluster_silhouette_values = sample_silhouette_values[clusters == cluster]
        cluster_silhouette_values.sort()

        cluster_size = cluster_silhouette_values.shape[0]
        y_upper = y_lower + cluster_size

        color = cm.nipy_spectral(float(cluster) / k1.elbow)
        ax.fill_betweenx(np.arange(y_lower, y_upper),0, cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)

        # Etichettiamo il cluster
        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(cluster))
        y_lower = y_upper + 10  # Aggiungiamo spazio vuoto tra i plot dei cluster


    ax.set_title("Silhouette plot dei cluster")
    ax.set_xlabel("Valore silhouette")
    ax.set_ylabel("Cluster")
    ax.axvline(x=sil_score, color="red", linestyle="--")
    plt.show()
    
    return k1.elbow


def plot_learning_curves(model, X, y, target_column, model_name):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10, scoring='accuracy')

    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Stampa i valori numerici della deviazione standard e della varianza
    print(
        f"{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}")

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    #Visualizza la curva di apprendimento
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()
    plt.show()

def visualizeMetricsGraphs(evaluation_results):
    models = list(evaluation_results.keys())

    # Creazione di un array numpy per ogni metrica
    accuracy = np.array([evaluation_results[clf]['test_accuracy'] for clf in models])
    precision = np.array([evaluation_results[clf]['test_precision'] for clf in models])
    recall = np.array([evaluation_results[clf]['test_recall'] for clf in models])
    f1 = np.array([evaluation_results[clf]['test_f1'] for clf in models])

    # Creazione del grafico a barre
    bar_width = 0.2
    index = np.arange(len(models))
    plt.bar(index, accuracy, bar_width, label='Accuracy')
    plt.bar(index + bar_width, precision, bar_width, label='Precision')
    plt.bar(index + 2 * bar_width, recall, bar_width, label='Recall')
    plt.bar(index + 3 * bar_width, f1, bar_width, label='F1')
    # Aggiunta di etichette e legenda
    plt.xlabel('Modelli')
    plt.ylabel('Punteggi')
    plt.title('Punteggio medio per ogni modello')
    plt.xticks(index + 1.5 * bar_width, models)
    plt.legend()

    # Visualizzazione del grafico
    plt.show()


def oversampling(dataset,target_column):
    dataset = dataset.dropna(subset=[target_column])
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X,y)
    dataset_idrico_r = pd.DataFrame(X_resampled, columns = X.columns)
    dataset_idrico_r[target_column] = y_resampled
    #print("Oversampling")
    return dataset_idrico_r
