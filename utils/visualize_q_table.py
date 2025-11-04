import argparse
import os
import pickle

import numpy as np

# Usiamo un import relativo per trovare common_code nello stesso pacchetto
from . import common_code as common


def visualize_q_table(filepath):
    """
    Carica una Q-table da un file .pkl e la stampa in un formato leggibile.
    Loads a Q-table from a .pkl file and prints it in a readable format.

    Args:
        filepath (str): Percorso del file .pkl da visualizzare.
    """
    if not os.path.exists(filepath):
        print(f"Errore: Il file '{filepath}' non è stato trovato.")
        return

    try:
        with open(filepath, "rb") as f:
            q_table = pickle.load(f)

        print(f"--- Visualizzazione della Q-Table da: {filepath} ---")
        print(f"Forma della tabella (Stati, Azioni): {q_table.shape}\n")

        # Imposta le opzioni di stampa di NumPy per una migliore leggibilità
        np.set_printoptions(precision=4, suppress=True)

        print(q_table)
        print("\n--- Fine della visualizzazione ---")

    except Exception as e:
        print(f"Si è verificato un errore durante la lettura del file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualizza una Q-Table salvata in un file .pkl."
    )
    parser.add_argument(
        "file",
        type=str,
        nargs="?",
        default=None,
        help="Percorso del file .pkl da visualizzare. Se omesso, verrà cercato il modello più recente.",
    )
    args = parser.parse_args()

    file_to_visualize = args.file

    if file_to_visualize is None:
        print(
            "Nessun file specificato. Ricerca del modello più recente in 'model_results'..."
        )
        # Chiama find_latest_model senza prefisso per trovare l'ultimo file in assoluto
        file_to_visualize = common.find_latest_model(
            directory="model_results", prefix=None
        )

    if file_to_visualize:
        visualize_q_table(file_to_visualize)
    else:
        print("Nessun modello trovato da visualizzare.")
