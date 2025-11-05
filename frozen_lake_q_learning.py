# frozen_lake_q_learning.py
import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np

from utils import common_code as common


def register_environment(max_episode_steps, reward_threshold):
    """
    Registra l'ambiente personalizzato FrozenLake-enhanced in Gymnasium.
    Registers the custom FrozenLake-enhanced environment in Gymnasium.
    """
    gym.register(
        id="FrozenLake-enhanced",
        entry_point="frozen_lake_enhanced:FrozenLakeEnv",
        kwargs={"map_name": "8x8"},
        max_episode_steps=max_episode_steps,
        reward_threshold=reward_threshold,
    )


def initialize_environment(render=False, is_slippery=True):
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
    steps = 0
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
        steps += 1
    if reward == 1:
        rewards_per_episode[episode] = 1

    epsilon = max(epsilon - 0.0001, 0)
    if epsilon == 0:
        learning_rate_a = 0.0001

    return q, epsilon, learning_rate_a, steps


def save_model(q, episodes, rewards_per_episode, prefix="frozen_lake"):
    """
    Salva il modello e i risultati del training.
    Saves the model and training results.
    """
    results_dir = "model_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    model_filename = f"{prefix}_{timestamp}.pkl"
    plot_filename = f"{prefix}_{timestamp}.png"
    model_filepath = os.path.join(results_dir, model_filename)

    common.plot_rewards(
        rewards_per_episode, episodes, os.path.join(results_dir, plot_filename)
    )
    common.save_q_table(q, model_filepath)
    print(f"Modello salvato in: {model_filepath}")
    return model_filepath, timestamp


def run_training(
    episodes=20000,
    learning_rate_a=0.9,
    discount_factor_g=0.9,
    epsilon=1,
    prefix="frozen_lake",
    max_episode_steps=100,
    reward_threshold=0.85,
    is_slippery=True,
):
    """
    Esegue il training dell'agente Q-Learning.
    Runs the Q-Learning agent training.
    Args:
        episodes (int): Numero di episodi per il training.
                        Number of episodes for training.
    """
    print("Modalità training selezionata.")
    register_environment(max_episode_steps, reward_threshold)
    env = initialize_environment(render=False, is_slippery=is_slippery)
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)
    q = common.initialize_q_table(env)

    for i in range(episodes):
        q, epsilon, learning_rate_a, _ = run_episode(
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
    model_filepath, timestamp = save_model(q, episodes, rewards_per_episode, prefix)

    # --- Fase di Valutazione Automatica ---
    print("\nAvvio della valutazione del modello appena addestrato...")
    eval_episodes = 100  # Numero di episodi per la valutazione
    eval_env = initialize_environment(
        render=False
    )  # Nessun rendering per la valutazione
    eval_rng = np.random.default_rng()
    eval_rewards = np.zeros(eval_episodes)
    eval_steps = []

    for i in range(eval_episodes):
        _, _, _, steps = run_episode(
            eval_env,
            q,  # Usa la Q-table appena addestrata
            0,  # Epsilon a 0 per una politica puramente greedy (sfruttamento)
            0,  # Learning rate non necessario
            discount_factor_g,
            eval_rng,
            False,  # Non è training
            i,
            eval_rewards,
        )
        eval_steps.append(steps)
    eval_env.close()

    # Calcola le metriche di qualità
    success_rate = (np.sum(eval_rewards) / eval_episodes) * 100
    avg_steps = np.mean(eval_steps)

    # --- Registrazione dei Risultati ---
    log_data = {
        "timestamp": timestamp,
        "model_file": os.path.basename(model_filepath),
        "episodes": episodes,
        "learning_rate": learning_rate_a,
        "discount_factor": discount_factor_g,
        "start_epsilon": epsilon,
        "max_episode_steps": max_episode_steps,
        "reward_threshold": reward_threshold,
        "success_rate_perc": f"{success_rate:.2f}",
        "avg_steps": f"{avg_steps:.2f}",
    }

    log_filepath = os.path.join("model_results", "training_log.csv")
    common.log_training_results(log_filepath, log_data)


def run_execution(
    episodes=50,
    learning_rate_a=0.9,
    discount_factor_g=0.9,
    epsilon=1,
    model_to_load=None,
    prefix="frozen_lake",
    max_episode_steps=100,
    reward_threshold=0.85,
    is_slippery=True,
):
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
        model_to_load = common.find_latest_model(prefix=prefix)
        print(
            f"Nessun file specificato. Caricamento del modello più recente: {model_to_load}"
        )

    register_environment(max_episode_steps, reward_threshold)
    env = initialize_environment(render=True, is_slippery=is_slippery)
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = []
    q = common.load_q_table(model_to_load)

    for i in range(episodes):
        q, epsilon, learning_rate_a, steps = run_episode(
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
        steps_per_episode.append(steps)

    env.close()
    print("\n--- Report di Valutazione del Modello ---")

    total_wins = np.sum(rewards_per_episode)
    total_episodes = len(rewards_per_episode)
    success_rate = (total_wins / total_episodes) * 100

    avg_steps = np.mean(steps_per_episode)

    print(f"Episodi di test eseguiti: {total_episodes}")
    print(f"Vittorie totali: {int(total_wins)}")
    print(f"Tasso di successo: {success_rate:.2f}%")
    print(f"Numero medio di passi per episodio: {avg_steps:.2f}")
    print("----------------------------------------")


if __name__ == "__main__":

    IS_SLIPPERY = False
    LEARNING_RATE_A = 0.1
    DISCOUNT_FACTOR_G = 0.9  # future reward discount
    EPSILON = 1  #  greedy (exploration Vs explotation)
    EPISODES = 20_000  # 1_000

    MAX_EPISODE_STEPS = 200
    PREFIX = "frozen_lake"
    REWARD_THRESHOLD = 0.95  # was 0.85

    args = common.parse_arguments()
    if args.mode == "train":
        run_training(
            episodes=EPISODES,
            learning_rate_a=LEARNING_RATE_A,
            discount_factor_g=DISCOUNT_FACTOR_G,
            epsilon=EPSILON,
            prefix=PREFIX,
            max_episode_steps=MAX_EPISODE_STEPS,
            reward_threshold=REWARD_THRESHOLD,
            is_slippery=IS_SLIPPERY,
        )
    elif args.mode == "exec":
        run_execution(
            model_to_load=args.file,
            learning_rate_a=LEARNING_RATE_A,
            discount_factor_g=DISCOUNT_FACTOR_G,
            epsilon=0,  # Attention!!
            prefix=PREFIX,
            max_episode_steps=MAX_EPISODE_STEPS,
            reward_threshold=REWARD_THRESHOLD,
            is_slippery=IS_SLIPPERY,
        )
