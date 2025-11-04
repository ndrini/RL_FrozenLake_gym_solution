# frozen_lake_q_learning.py
import argparse
import os
import pickle
from datetime import datetime

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from utils import common_code as common


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
    """
    state = env.reset()[0]
    terminated = False
    truncated = False
    reward = 0  # Inizializza reward
    while not terminated and not truncated:
        action = common.select_action(state, q, epsilon, rng, env, is_training)
        new_state, reward, terminated, truncated, _ = env.step(action)
        if is_training:
            q = common.update_q_table(
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


def save_model(q, episodes, rewards_per_episode):
    """
    Salva il modello e i risultati del training.
    Saves the model and training results.
    """
    results_dir = "model_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    base_filepath = os.path.join(results_dir, f"q_table_{timestamp}")
    common.plot_rewards(rewards_per_episode, episodes, f"{base_filepath}.png")
    common.save_q_table(q, f"{base_filepath}.pkl")
    print(f"Modello salvato in: {base_filepath}.pkl")


def find_latest_model():
    """
    Trova il file del modello più recente nella cartella model_results.
    Finds the latest model file in the model_results directory.
    Returns:
        str: Percorso del file del modello più recente.
             Path to the latest model file.
    """
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
    return max(files, key=os.path.getctime)


def run_training(episodes=20000):
    """
    Esegue il training dell'agente Q-Learning.
    Runs the Q-Learning agent training.
    Args:
        episodes (int): Numero di episodi per il training.
                        Number of episodes for training.
    """
    print("Modalità training selezionata.")
    register_environment()
    env = initialize_environment(render=False)
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)
    q = common.initialize_q_table(env)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1

    for i in range(episodes):
        q, epsilon, learning_rate_a = run_episode(
            env,
            q,
            epsilon,
            learning_rate_a,
            discount_factor_g,
            rng,
            True,
            i,
            rewards_per_episode,
        )

    env.close()
    save_model(q, episodes, rewards_per_episode)


def run_execution(episodes=50, model_to_load=None):
    """
    Esegue l'agente Q-Learning con un modello pre-addestrato.
    Runs the Q-Learning agent with a pre-trained model.
    Args:
        episodes (int): Numero di episodi da eseguire.
                        Number of episodes to run.
        model_to_load (str): Percorso del file del modello da caricare.
                             Path to the model file to load.
    """
    print("Modalità esecuzione selezionata.")
    if model_to_load is None:
        model_to_load = find_latest_model()
        print(
            f"Nessun file specificato. Caricamento del modello più recente: {model_to_load}"
        )

    register_environment()
    env = initialize_environment(render=True)
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)
    q = common.load_q_table(model_to_load)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 0  # In modalità esecuzione, non vogliamo esplorazione casuale

    for i in range(episodes):
        q, epsilon, learning_rate_a = run_episode(
            env,
            q,
            epsilon,
            learning_rate_a,
            discount_factor_g,
            rng,
            False,
            i,
            rewards_per_episode,
        )

    env.close()


def main():
    """
    Funzione principale.
    Main function.
    """
    args = common.parse_arguments()
    if args.mode == "train":
        run_training(episodes=200)
    elif args.mode == "exec":
        run_execution(model_to_load=args.file)


if __name__ == "__main__":
    main()
