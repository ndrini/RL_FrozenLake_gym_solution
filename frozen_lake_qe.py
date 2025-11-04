# This file is almost identical to frozen_lake_q.py, except this uses the frozen_lake_enhanced.py environment.

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gymnasium.spaces import Discrete


# Register the enhanced frozen lake environment
# Sample of registration entry found in C:\Users\<username>\.conda\envs\gymenv\Lib\site-packages\gymnasium\envs\__init__.py
gym.register(
    id="FrozenLake-enhanced", # give it a unique id
    entry_point="frozen_lake_enhanced:FrozenLakeEnv", # frozen_lake_enhanced = name of file 'frozen_lake_enhanced.py'
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,  # optimum = 0.91
)

"""
gym.register(...): Dice a Gymnasium: "Ehi, ho creato un ambiente personalizzato! Se qualcuno chiede di creare un ambiente con l'ID 'FrozenLake-enhanced', tu devi andare a cercare la classe FrozenLakeEnv nel file frozen_lake_enhanced.py".
max_episode_steps=200: Imposta un limite di 200 azioni per ogni partita. Se l'agente non raggiunge l'obiettivo o non cade in un buco entro 200 passi, la partita finisce (questo si chiama truncation).

id: Un identificatore univoco per l'ambiente.
entry_point: Il percorso al file Python che definisce l'ambiente personalizzato. In questo caso, il file si chiama frozen_lake_enhanced.py e contiene la classe FrozenLakeEnv.
kwargs: Argomenti aggiuntivi per la creazione dell'ambiente. Qui, map_name="8x8" specifica che la mappa del labirinto è di dimensioni 8x8.
max_episode_steps: Il numero massimo di passi per episodio. Dopo 200 passi, l'episodio viene interrotto.
reward_threshold: La soglia di ricompensa considerata ottimale. In questo caso, una politica che ottiene una ricompensa media di 0.85 è considerata ottimale.

"""


def run(episodes, is_training=True, render=False):
    """ 
    """

    # 'FrozenLake-enhanced' is the id specified above
    env = gym.make('FrozenLake-enhanced', desc=None, map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):  # training mode
        if isinstance(env.observation_space, Discrete) and isinstance(env.action_space, Discrete):
            q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 × 4 array
        else: # type: ignore
            raise ValueError("Observation space is not discrete. Q-learning requires a discrete observation space.")
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10_000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            # pass the q table and episode count to the environment for rendering
            if(env.render_mode=='human'):
                env.unwrapped.set_q(q)
                env.unwrapped.set_episode(i)

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    """
    7. Visualizzazione e Salvataggio dei Risultati
    Alla fine dell'apprendimento, i risultati vengono visualizzati e salvati:
    """
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':

    run(50, is_training=False, render=True)
