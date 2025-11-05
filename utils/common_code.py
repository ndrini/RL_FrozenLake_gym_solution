import argparse
import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Discrete


def initialize_q_table(env):
    """
    Inizializza la tabella Q per l'ambiente dato.
    Initializes the Q-table for the given environment.

    Args:
        env: L'ambiente Gymnasium.
             The Gymnasium environment.

    Returns:
        q (np.ndarray): La tabella Q inizializzata.
                         The initialized Q-table.
    """
    if isinstance(env.observation_space, Discrete) and isinstance(
        env.action_space, Discrete
    ):
        return np.zeros((env.observation_space.n, env.action_space.n))
    else:
        raise ValueError(
            "Observation/Action space is not discrete. Q-learning requires discrete spaces."
        )


def load_q_table(filepath):
    """
    Carica la tabella Q da un file.
    Loads the Q-table from a file.

    Returns:
        q (np.ndarray): La tabella Q caricata.
                         The loaded Q-table.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_q_table(q, filepath):
    """
    Salva la tabella Q su un file.
    Saves the Q-table to a file.

    Args:
        q (np.ndarray): La tabella Q da salvare.
                         The Q-table to save.
        filepath (str): Il percorso del file dove salvare la tabella.
                        The filepath where to save the table.
    """
    with open(filepath, "wb") as f:
        pickle.dump(q, f)


def select_action(state, q, epsilon, rng, env, is_training):
    """
    Seleziona un'azione usando una politica epsilon-greedy.
    Selects an action using an epsilon-greedy policy.

    Args:
        state: Lo stato corrente.
                The current state.
        q (np.ndarray): La tabella Q.
                         The Q-table.
        epsilon (float): Probabilità di scegliere un'azione casuale.
                          Probability of choosing a random action.
        rng (np.random.Generator): Generatore di numeri casuali.
                                    Random number generator.
        env: L'ambiente Gymnasium.
             The Gymnasium environment.
        is_training (bool): Se True, l'agente è in fase di addestramento.
                           If True, the agent is in training mode.

    Returns:
        action: L'azione selezionata.
                 The selected action.
    """
    if is_training and rng.random() < epsilon:
        return env.action_space.sample()  # Azione casuale
    else:
        return np.argmax(q[state, :])  # Azione con il valore Q più alto


def update_q_table(
    q, state, action, reward, new_state, learning_rate_a, discount_factor_g
):
    """
    Aggiorna la tabella Q usando l'equazione di Bellman.
    Updates the Q-table using the Bellman equation.
    """
    q[state, action] = q[state, action] + learning_rate_a * (
        reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
    )
    return q


def plot_rewards(rewards_per_episode, episodes, filepath):
    """
    Plotta la somma delle ricompense per episodio.
    Plots the sum of rewards per episode.

    Args:
        rewards_per_episode (np.ndarray): Array delle ricompense per episodio.
                                           Array of rewards per episode.
        episodes (int): Numero totale di episodi.
                         Total number of episodes.
        filepath (str): Percorso dove salvare il grafico.
                        Filepath to save the plot.
    """
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100) : (t + 1)])
    plt.plot(sum_rewards)
    plt.savefig(filepath)


def parse_arguments():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Esegue il training o il test dell'agente Q-Learning per FrozenLake."
    )
    parser.add_argument(
        "mode",
        choices=["train", "exec"],
        help="Modalità di esecuzione: 'train' per addestrare un nuovo modello, 'exec' per eseguire un modello salvato.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="[Solo per 'exec'] Percorso del file .pkl da caricare. Se non specificato, carica il più recente.",
    )
    return parser.parse_args()


def find_latest_model(directory="model_results", prefix=None):
    """
    Trova il file del modello (.pkl) più recente in una cartella, opzionalmente filtrando per prefisso.
    Finds the most recent model file (.pkl) in a directory, optionally filtering by a prefix.

    Args:
        directory (str): La cartella in cui cercare.
                         The directory to search in.
        prefix (str, optional): Prefisso per identificare i file (es. 'frozen_lake').
                                Prefix to identify files (e.g., 'frozen_lake').

    Returns:
        str: Il percorso del file più recente, o None se non trovato.
             The path to the latest file, or None if not found.
    """
    if not os.path.isdir(directory) or not os.listdir(directory):
        print(
            f"Errore: La cartella '{directory}' è vuota o non esiste. Esegui prima il training."
        )
        return None

    files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pkl")
    ]
    if prefix:
        files = [f for f in files if os.path.basename(f).startswith(prefix)]

    if not files:
        print(
            f"Errore: Nessun file .pkl con prefisso '{prefix}' trovato in '{directory}'."
        )
        return None

    return max(files, key=os.path.getctime)


def log_training_results(log_filepath, data_dict):
    """
    Registra i risultati di un'esecuzione di training in un file CSV.
    Logs the results of a training run to a CSV file.

    Args:
        log_filepath (str): Percorso del file CSV di log.
        data_dict (dict): Dizionario contenente i dati da registrare.
    """
    # Controlla se il file esiste per decidere se scrivere l'intestazione
    file_exists = os.path.isfile(log_filepath)

    try:
        with open(log_filepath, mode="a", newline="", encoding="utf-8") as csvfile:
            # Le chiavi del dizionario saranno le intestazioni delle colonne
            fieldnames = data_dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Scrive l'intestazione solo se il file è nuovo
            if not file_exists:
                writer.writeheader()

            # Scrive la riga di dati
            writer.writerow(data_dict)

        print(f"Risultati del training registrati in: {log_filepath}")

    except IOError as e:
        print(f"Errore durante la scrittura del file di log: {e}")
