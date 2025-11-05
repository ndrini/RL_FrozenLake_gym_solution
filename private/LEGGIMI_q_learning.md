# Q-learning


Vedi un esempio di codice in [questo file](../frozen_lake_q_learning.py).



## La bontà dell'allenamento 

In RL posso trovare alcuni  parametri sintetici  che indicano  la qualità dell'allenamento di un modello, ma non esiste un singolo parametro "universale" valido per ogni problema,


Per un ambiente come FrozenLake, ci sono alcuni indicatori sintetici molto efficaci che ti dicono chiaramente la qualità di un modello addestrato: Success Rate / Win Rate.



### Tasso di Successo (Success Rate / Win Rate):


Tasso di Successo (Success Rate / Win Rate):
- Cos'è: La percentuale di episodi in cui l'agente riesce a raggiungere l'obiettivo (G) senza cadere in un buco (H).
- Perché è utile: Per FrozenLake, è la metrica più intuitiva. Un valore del 75% significa che, su 100 tentativi, l'agente ha risolto il labirinto 75 volte. È un indicatore diretto della sua affidabilità.


Media delle Ricompense (Average Reward):

- Cos'è: La ricompensa media ottenuta per episodio, calcolata su un certo numero di esecuzioni di test. Poiché in questo ambiente la ricompensa è +1 solo per la vittoria, questa metrica è strettamente legata al tasso di successo.
- Perché è utile: È una metrica standard in tutto il Reinforcement Learning. In ambienti più complessi con ricompense variabili, diventa l'indicatore principale.

Average Steps per Episode
- Un'altra metrica utile è la Media dei Passi per Episodio (Average Steps per Episode), che misura l'efficienza dell'agente (un buon agente non solo vince, ma lo fa nel minor numero di passi possibile).


### Visualizzazione dei risultati


Con i comando 


```bash
RL_FrozenLake_gym_solution$ python -m utils.visualize_historial

RL_FrozenLake_gym_solution$ python -m utils.visualize_q_table


```


## i parametri con cui "giocare"



### fondamentele: non stocastico 

Soluzione 4 — FrozenLake è “slippery”  False


### parametri sulla velocità di apprendimento

Ti propongo una cosa pulita, semplice e stabile:

1) Usiamo solo decadimento esponenziale di epsilon
2) 
Hai:

# Aggiornamento epsilon (solo esponenziale)
epsilon = max(0.01, epsilon * 0.995)

# Learning rate che si adatta gradualmente
learning_rate_a = max(0.1 * epsilon, 0.0001)

