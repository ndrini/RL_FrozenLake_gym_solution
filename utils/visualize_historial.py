import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Usiamo un import relativo per trovare common_code nello stesso pacchetto
from . import common_code as common


def visualize_historial(log_filepath):
    """
    Carica lo storico dei training da un file CSV e genera visualizzazioni significative.
    Loads the training history from a CSV file and generates meaningful visualizations.

    Args:
        log_filepath (str): Percorso del file CSV di log.
    """
    if not os.path.exists(log_filepath):
        print(f"Errore: Il file di log '{log_filepath}' non è stato trovato.")
        return

    try:
        # Carica i dati usando pandas
        df = pd.read_csv(log_filepath)
        print("Dati caricati con successo. Colonne disponibili:", df.columns.tolist())

        # --- Grafico 1: Tasso di Successo vs. Numero di Episodi ---
        # Questo grafico è fondamentale per capire se l'agente impara nel tempo.
        plt.figure(figsize=(12, 7))
        sns.scatterplot(
            data=df,
            x="episodes",
            y="success_rate_perc",
            hue="learning_rate",  # Il colore indica il learning rate
            size="discount_factor",  # La dimensione indica il discount factor
            palette="viridis",
            sizes=(50, 250),
            legend="auto",
        )
        plt.title(
            "Tasso di Successo vs. Episodi di Training", fontsize=16, fontweight="bold"
        )
        plt.xlabel("Numero di Episodi", fontsize=12)
        plt.ylabel("Tasso di Successo (%)", fontsize=12)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

        # --- Grafico 2: Heatmap degli Iperparametri ---
        # Questa heatmap mostra quali combinazioni di learning_rate e discount_factor
        # portano a un tasso di successo maggiore.
        if df["learning_rate"].nunique() > 1 and df["discount_factor"].nunique() > 1:
            # Aggreghiamo i dati per ottenere il massimo tasso di successo per ogni combinazione
            pivot_table = df.pivot_table(
                values="success_rate_perc",
                index="learning_rate",
                columns="discount_factor",
                aggfunc="max",  # Mostra il miglior risultato ottenuto
            )

            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_table,
                annot=True,  # Mostra i valori numerici nelle celle
                fmt=".1f",  # Formato a una cifra decimale
                cmap="YlGnBu",
                linewidths=0.5,
            )
            plt.title(
                "Heatmap: Tasso di Successo Massimo (%) per Iperparametri",
                fontsize=16,
                fontweight="bold",
            )
            plt.xlabel("Discount Factor (gamma)", fontsize=12)
            plt.ylabel("Learning Rate (alpha)", fontsize=12)
            plt.show()

    except Exception as e:
        print(f"Si è verificato un errore durante la creazione dei grafici: {e}")


if __name__ == "__main__":
    # Il file di log si trova nella cartella model_results
    log_file = os.path.join("model_results", "training_log.csv")
    visualize_historial(log_file)
