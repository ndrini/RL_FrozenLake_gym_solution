# frozen_lake_q_learning.py

import argparse
import os
import pickle
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Discrete


def register_environment():
    """
    Registra l'ambiente personalizzato FrozenLake-enhanced in Gymnasium.
    Registers the custom FrozenLake-enhanced environment in Gymnasium.
    """
    gym.register(
        id="FrozenLake-enhanced",
        entry_point="frozen_lake_enhanced:FrozenLakeEnv",
        kwargs={"map_name": "8x8"},
        max_episode_steps=200,
        reward_threshold=0.85,
    )


def initialize_environment(render=False):
    """
    Inizializza e restituisce l'ambiente FrozenLake-enhanced.
    Initializes and returns the FrozenLake-enhanced environment.

    Args:
        render (bool): Se True, abilita la modalità di rendering.
                       If True, enables rendering mode.

    Returns:
        env: L'ambiente Gymnasium inizializzato.
             The initialized Gymnasium environment.
    """
    env = gym.make(
        "FrozenLake-enhanced",
        desc=None,
        map_name="8x8",
        is_slippery=True,
        render_mode="human" if render else None,
    )
    return env


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
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        raise ValueError(
            "Observation space is not discrete. Q-learning requires a discrete observation space."
        )
    return q


def load_q_table(filepath):
    """
    Carica la tabella Q da un file.
    Loads the Q-table from a file.

    Returns:
        q (np.ndarray): La tabella Q caricata.
                         The loaded Q-table.
    """
    with open(filepath, "rb") as f:
        q = pickle.load(f)
    return q


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
        action = env.action_space.sample()  # Azione casuale
    else:
        action = np.argmax(q[state, :])  # Azione con il valore Q più alto
    return action


def update_q_table(
    q, state, action, reward, new_state, learning_rate_a, discount_factor_g
):
    """
    Aggiorna la tabella Q usando l'equazione di Bellman.
    Updates the Q-table using the Bellman equation.

    Args:
        q (np.ndarray): La tabella Q.
                         The Q-table.
        state: Lo stato corrente.
                The current state.
        action: L'azione eseguita.
                 The action taken.
        reward (float): La ricompensa ottenuta.
                         The reward received.
        new_state: Il nuovo stato dopo l'azione.
                    The new state after the action.
        learning_rate_a (float): Tasso di apprendimento.
                                  Learning rate.
        discount_factor_g (float): Fattore di sconto.
                                    Discount factor.

    Returns:
        q (np.ndarray): La tabella Q aggiornata.
                         The updated Q-table.
    """
    q[state, action] = q[state, action] + learning_rate_a * (
        reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
    )
    return q


def run_episode(
    env,
    q,
    epsilon,
    learning_rate_a,
    discount_factor_g,
    rng,
    is_training,
    episode,
    rewards_per_episode,
):
    """
    Esegue un singolo episodio.
    Runs a single episode.

    Args:
        env: L'ambiente Gymnasium.
             The Gymnasium environment.
        q (np.ndarray): La tabella Q.
                         The Q-table.
        epsilon (float): Probabilità di scegliere un'azione casuale.
                          Probability of choosing a random action.
        learning_rate_a (float): Tasso di apprendimento.
                                  Learning rate.
        discount_factor_g (float): Fattore di sconto.
                                    Discount factor.
        rng (np.random.Generator): Generatore di numeri casuali.
                                    Random number generator.
        is_training (bool): Se True, l'agente è in fase di addestramento.
                           If True, the agent is in training mode.
        episode (int): Numero dell'episodio corrente.
                        Current episode number.
        rewards_per_episode (np.ndarray): Array per memorizzare le ricompense per episodio.
                                           Array to store rewards per episode.

    Returns:
        q (np.ndarray): La tabella Q aggiornata.
                         The updated Q-table.
        epsilon (float): Il valore aggiornato di epsilon.
                          The updated epsilon value.
        learning_rate_a (float): Il valore aggiornato del tasso di apprendimento.
                                  The updated learning rate value.
    """
    state = env.reset()[0]
    terminated = False
    truncated = False
    reward = 0  # Inizializza reward
    while not terminated and not truncated:
        action = select_action(state, q, epsilon, rng, env, is_training)
        new_state, reward, terminated, truncated, _ = env.step(action)
        if is_training:
            q = update_q_table(
                q, state, action, reward, new_state, learning_rate_a, discount_factor_g
            )
        if env.render_mode == "human":
            env.unwrapped.set_q(q)
            env.unwrapped.set_episode(episode)
        state = new_state
    if reward == 1:
        rewards_per_episode[episode] = 1
    epsilon = max(epsilon - 0.0001, 0)
    if epsilon == 0:
        learning_rate_a = 0.0001
    return q, epsilon, learning_rate_a


def plot_rewards(rewards_per_episode, episodes, filepath):
    """
    Plotta la somma delle ricompense per episodio.
    Plots the sum of rewards per episode.

    Args:
        rewards_per_episode (np.ndarray): Array delle ricompense per episodio.
                                           Array of rewards per episode.
        episodes (int): Numero totale di episodi.
                         Total number of episodes.
    """
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100) : (t + 1)])
    plt.plot(sum_rewards)
    plt.savefig(filepath)


def run(
    episodes,
    is_training=True,
    render=False,
    learning_rate_a=0.9,
    discount_factor_g=0.9,
    epsilon=1,
    load_file=None,
):
    """
    Esegue l'algoritmo di Q-Learning per un dato numero di episodi.
    Runs the Q-Learning algorithm for a given number of episodes.

    Args:
        episodes (int): Numero di episodi da eseguire.
                         Number of episodes to run.
        is_training (bool): Se True, l'agente è in fase di addestramento.
                           If True, the agent is in training mode.
        render (bool): Se True, abilita la modalità di rendering.
                      If True, enables rendering mode.
        load_file (str, optional): Percorso del file del modello da caricare.
                                   Path to the model file to load.
    """
    register_environment()
    env = initialize_environment(render)
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    if is_training:
        q = initialize_q_table(env)
    else:
        if load_file is None:
            raise ValueError(
                "È necessario specificare un file da caricare in modalità esecuzione."
            )
        print(f"Caricamento del modello da: {load_file}")
        q = load_q_table(load_file)

    for i in range(episodes):
        q, epsilon, learning_rate_a = run_episode(
            env,
            q,
            epsilon,
            learning_rate_a,
            discount_factor_g,
            rng,
            is_training,
            i,
            rewards_per_episode,
        )

    env.close()
    if is_training:
        # Crea la cartella dei risultati se non esiste
        results_dir = "model_results"
        os.makedirs(results_dir, exist_ok=True)

        # Genera il percorso del file con timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        base_filepath = os.path.join(results_dir, f"q_table_{timestamp}")
        plot_rewards(rewards_per_episode, episodes, f"{base_filepath}.png")
        save_q_table(q, f"{base_filepath}.pkl")


if __name__ == "__main__":
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

    args = parser.parse_args()

    if args.mode == "train":
        print("Modalità training selezionata.")
        run(episodes=20000, is_training=True, render=False)
    elif args.mode == "exec":
        print("Modalità esecuzione selezionata.")
        model_to_load = args.file

        if model_to_load is None:
            # Trova il file più recente nella cartella model_results
            results_dir = "model_results"
            if not os.path.isdir(results_dir) or not os.listdir(results_dir):
                print(
                    f"Errore: La cartella '{results_dir}' è vuota o non esiste. Esegui prima il training."
                )
                exit()

            files = [
                os.path.join(results_dir, f)
                for f in os.listdir(results_dir)
                if f.endswith(".pkl")
            ]
            if not files:
                print(f"Errore: Nessun file .pkl trovato in '{results_dir}'.")
                exit()

            model_to_load = max(files, key=os.path.getctime)
            print(
                f"Nessun file specificato. Caricamento del modello più recente: {model_to_load}"
            )

        run(episodes=50, is_training=False, render=True, load_file=model_to_load)
