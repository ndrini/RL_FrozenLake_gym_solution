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
