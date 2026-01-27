import time

from src.ai.ql_agent import QLearningAgent
from src.training.gym import train_with_decay

NUM_EPISODES = 50000

if __name__ == "__main__":
    inicio = time.time()
    alpha = 0.45
    gamma = 0.38
    decay = 0.06
    r_draw = 0.50
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=1.0)

    agent, _ = train_with_decay(agent, episodes=NUM_EPISODES, epsilon_decay_gen=decay, reward_draw_gen=r_draw)
    fin = time.time()
    print(fin - inicio)
