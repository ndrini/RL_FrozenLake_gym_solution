

 
# frozen_lake_q_learning.py

import argparse
import pickle

import gymnasium as gym


if __name__ == "__main__":
    # run(
    #     10000,
    #     is_training=True,
    #     render=False,
    #     learning_rate_a=0.9,
    #     discount_factor_g=0.9,
    #     epsilon=1,
    # )
    run(
        50,
        is_training=False,
        render=True,
    )
    parser = argparse.ArgumentParser(
        description="Esegue il training o il test dell'agente Q-Learning per FrozenLake."
    )
    parser.add_argument(
        "mode",
        choices=["train", "exec"],
        help="Modalità di esecuzione: 'train' per addestrare un nuovo modello, 'exec' per eseguire un modello salvato.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        print("Modalità training selezionata.")
        # Esegue il training per 15000 episodi senza rendering per velocizzare
        run(episodes=15000, is_training=True, render=False)
    elif args.mode == "exec":
        print("Modalità esecuzione selezionata.")
        # Esegue 50 episodi di test con rendering per mostrare il risultato
        run(episodes=50, is_training=False, render=True)

Come Usarlo Ora
Dopo aver applicato queste modifiche, potrai lanciare lo script dal tuo terminale in questi modi:

Per avviare il training:

bash
python frozen_lake_q_learning.py train
Output atteso:

plaintext
Modalità training selezionata.
(Il training inizierà, senza finestra grafica, e alla fine salverà il modello e il grafico)
Per eseguire e testare un modello già addestrato:

bash
python frozen_lake_q_learning.py exec
Output atteso:

plaintext
Modalità esecuzione selezionata.
(Si aprirà la finestra grafica e l'agente giocherà per 50 partite usando il file frozen_lake8x8.pkl)
Per vedere il messaggio di aiuto (generato automaticamente da argparse):

bash
python frozen_lake_q_learning.py -h
Output atteso:

plaintext
 Show full code block 
usage: frozen_lake_q_learning.py [-h] {train,exec}

Esegue il training o il test dell'agente Q-Learning per FrozenLake.

positional arguments:
  {train,exec}  Modalità di esecuzione: 'train' per addestrare un nuovo modello, 'exec' per eseguire un modello salvato.

options:
  -h, --help      show this help message and exit
Questo refactoring rende il tuo 